from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.pipeline.types import Segment


def assign_two_segment_speakers(
    segments: list[Segment],
    usable_indices: list[int],
    embeddings: list[object],
    *,
    threshold: float = 0.18,
) -> None:
    import numpy as np

    stacked = np.vstack(embeddings)
    norms = np.linalg.norm(stacked, axis=1)
    if np.any(norms == 0):
        return
    cosine = float(np.dot(stacked[0], stacked[1]) / (norms[0] * norms[1]))
    if 1 - cosine >= threshold:
        for speaker_number, segment_index in enumerate(usable_indices, start=1):
            segments[segment_index].speaker = f"Speaker {speaker_number}"


class Diarizer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def assign_speakers(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        if self.settings.engine_mode == "demo":
            return segments
        try:
            assigned = self._speechbrain_cluster(audio_path, segments)
        except Exception:
            # Diarization is explicitly best-effort. Keeping the transcript usable is
            # more valuable than failing the whole job when embeddings are unavailable.
            assigned = segments
        return self._short_dialogue_fallback(assigned)

    def _speechbrain_cluster(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        import numpy as np
        import torch
        import torchaudio
        from sklearn.cluster import AgglomerativeClustering
        from speechbrain.inference.speaker import EncoderClassifier

        if len(segments) < 2:
            return segments

        signal, sample_rate = torchaudio.load(str(audio_path))
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(self.settings.storage_root / "models" / "speechbrain_ecapa"),
        )

        embeddings: list[np.ndarray] = []
        usable_indices: list[int] = []
        for index, segment in enumerate(segments):
            start = max(0, int(segment.start * sample_rate))
            end = min(signal.shape[1], int(segment.end * sample_rate))
            if end - start < sample_rate // 2:
                continue
            clip = signal[:, start:end]
            with torch.no_grad():
                embedding = classifier.encode_batch(clip).squeeze().cpu().numpy()
            embeddings.append(embedding)
            usable_indices.append(index)

        if len(embeddings) < 2:
            return segments
        if len(embeddings) == 2:
            assign_two_segment_speakers(segments, usable_indices, embeddings)
            return segments

        # Conservative estimate: short clips normally have one or two speakers.
        n_clusters = 2 if len(embeddings) < 12 else min(4, max(2, len(embeddings) // 8))
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(np.vstack(embeddings))
        for segment_index, label in zip(usable_indices, labels):
            segments[segment_index].speaker = f"Speaker {int(label) + 1}"
        return segments

    def _short_dialogue_fallback(self, segments: list[Segment]) -> list[Segment]:
        if len(segments) != 2:
            return segments
        if segments[0].speaker != segments[1].speaker:
            return segments
        # shortcut: two-turn interview clips often lack enough segments for stable clustering.
        if segments[1].start - segments[0].end >= 0.25:
            segments[0].speaker = "Speaker 1"
            segments[1].speaker = "Speaker 2"
        return segments
