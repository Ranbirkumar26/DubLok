from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.pipeline.types import Segment

logger = logging.getLogger(__name__)

# ECAPA cosine distance: one speaker's turns measured up to ~0.42 on short clips,
# two different speakers ~0.94. 0.18 split single speakers.
SPEAKER_DISTANCE_THRESHOLD = 0.5


def assign_two_segment_speakers(
    segments: list[Segment],
    usable_indices: list[int],
    embeddings: list[object],
    *,
    threshold: float = SPEAKER_DISTANCE_THRESHOLD,
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
            return self._speechbrain_cluster(audio_path, segments)
        except Exception as exc:
            # Diarization is explicitly best-effort. Keeping the transcript usable is
            # more valuable than failing the whole job when embeddings are unavailable.
            # The pause heuristic only runs here: it must not override an embedding
            # decision that both turns are the same voice.
            logger.warning("Speaker embeddings unavailable, using pause heuristic: %s", exc)
            return self._short_dialogue_fallback(segments)

    def _speechbrain_cluster(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        import numpy as np
        import torch
        import soundfile
        from sklearn.cluster import AgglomerativeClustering
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        if len(segments) < 2:
            return segments

        # torchaudio.load needs torchcodec on torchaudio>=2.9; soundfile reads WAV directly.
        samples, sample_rate = soundfile.read(str(audio_path), dtype="float32", always_2d=True)
        signal = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(self.settings.storage_root / "models" / "speechbrain_ecapa"),
            # Default SYMLINK strategy needs admin/developer mode on Windows.
            local_strategy=LocalStrategy.COPY,
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

        # Cluster by voice distance instead of forcing a speaker count, so a
        # single-speaker clip stays one speaker.
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=SPEAKER_DISTANCE_THRESHOLD,
            metric="cosine",
            linkage="average",
        ).fit_predict(np.vstack(embeddings))
        numbering: dict[int, int] = {}
        for segment_index, label in zip(usable_indices, labels):
            number = numbering.setdefault(int(label), len(numbering) + 1)
            segments[segment_index].speaker = f"Speaker {number}"
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
