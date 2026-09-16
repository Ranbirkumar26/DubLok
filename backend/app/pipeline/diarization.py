from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.pipeline.types import Segment


class Diarizer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def assign_speakers(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        if self.settings.engine_mode == "demo":
            return segments
        try:
            return self._speechbrain_cluster(audio_path, segments)
        except Exception:
            # Diarization is explicitly best-effort. Keeping the transcript usable is
            # more valuable than failing the whole job when embeddings are unavailable.
            return segments

    def _speechbrain_cluster(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        import numpy as np
        import torch
        import torchaudio
        from sklearn.cluster import AgglomerativeClustering
        from speechbrain.inference.speaker import EncoderClassifier

        if len(segments) < 3:
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

        if len(embeddings) < 3:
            return segments

        # Conservative estimate: short clips normally have one or two speakers.
        n_clusters = 2 if len(embeddings) < 12 else min(4, max(2, len(embeddings) // 8))
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(np.vstack(embeddings))
        for segment_index, label in zip(usable_indices, labels):
            segments[segment_index].speaker = f"Speaker {int(label) + 1}"
        return segments
