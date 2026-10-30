from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.pipeline.types import Segment

logger = logging.getLogger(__name__)

# ECAPA cosine distance: one speaker's turns measured up to ~0.42 on short clips,
# two different speakers ~0.94. 0.18 split single speakers.
SPEAKER_DISTANCE_THRESHOLD = 0.5
# Embeddings of clips under ~1.5 s are unreliable (a 1.2 s line of one speaker was labelled as a
# second speaker), so shorter segments take the voice of the nearest longer neighbour instead.
MIN_EMBED_SECONDS = 1.5


def inherit_short_segment_speakers(segments: list[Segment], usable_indices: list[int]) -> None:
    usable = set(usable_indices)
    for index, segment in enumerate(segments):
        if index in usable:
            continue
        before = [i for i in usable_indices if i < index]
        after = [i for i in usable_indices if i > index]
        source = before[-1] if before else (after[0] if after else None)
        if source is not None:
            segment.speaker = segments[source].speaker


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


Turn = tuple[float, float, str]
PYANNOTE_PIPELINE = "pyannote/speaker-diarization-community-1"


def split_segments_by_turns(segments: list[Segment], turns: list[Turn]) -> list[Segment]:
    """Give every word its speaker from overlapping diarization turns.

    Turns may overlap in time. A word inside two speakers' turns belongs to the speaker of the
    previous word when that speaker is one of them (a speaker keeps the floor), else to the
    first listed. Segments split where the speaker changes, and a segment holding a word with
    more than one active speaker is flagged ``overlap``. Speakers are renumbered by first
    appearance. A lone word never forms its own segment.
    """
    ordered = sorted(turns)
    names: dict[str, str] = {}
    for _, _, label in ordered:
        names.setdefault(label, f"Speaker {len(names) + 1}")

    def active(moment: float) -> list[str]:
        found = [names[label] for start, end, label in ordered if start <= moment <= end]
        return list(dict.fromkeys(found))

    def nearest(moment: float) -> str | None:
        if not ordered:
            return None
        start, end, label = min(
            ordered, key=lambda turn: 0.0 if turn[0] <= moment <= turn[1] else min(abs(turn[0] - moment), abs(turn[1] - moment))
        )
        return names[label]

    result: list[Segment] = []
    last_speaker: str | None = None
    for segment in segments:
        if not segment.words:
            share: dict[str, float] = {}
            for start, end, label in ordered:
                length = min(end, segment.end) - max(start, segment.start)
                if length > 0:
                    share[names[label]] = share.get(names[label], 0.0) + length
            speaker = max(share, key=share.get) if share else (nearest(segment.start) or segment.speaker)
            result.append(Segment.from_dict({**segment.to_dict(), "speaker": speaker, "words": None}))
            last_speaker = speaker
            continue
        groups: list[dict] = []
        for word in segment.words:
            moment = (word["start"] + word["end"]) / 2
            candidates = active(moment)
            if len(candidates) > 1:
                speaker = last_speaker if last_speaker in candidates else candidates[0]
            else:
                speaker = candidates[0] if candidates else (last_speaker or nearest(moment) or segment.speaker)
            last_speaker = speaker
            if groups and groups[-1]["speaker"] == speaker:
                groups[-1]["words"].append(word)
            else:
                groups.append({"speaker": speaker, "words": [word], "overlap": False})
            if len(candidates) > 1:
                groups[-1]["overlap"] = True
        merged: list[dict] = []
        for group in groups:
            if merged and len(group["words"]) < 2:
                merged[-1]["words"] += group["words"]
                merged[-1]["overlap"] = merged[-1]["overlap"] or group["overlap"]
            else:
                merged.append(group)
        if len(merged) > 1 and len(merged[0]["words"]) < 2:
            merged[1]["words"] = merged[0]["words"] + merged[1]["words"]
            merged[1]["overlap"] = merged[1]["overlap"] or merged[0]["overlap"]
            merged.pop(0)
        for group in merged:
            words = group["words"]
            result.append(
                Segment(
                    id=segment.id,
                    speaker=group["speaker"],
                    start=float(words[0]["start"]),
                    end=float(words[-1]["end"]),
                    text="".join(w["word"] for w in words).strip(),
                    overlap=group["overlap"],
                )
            )
    for index, segment in enumerate(result, start=1):
        segment.id = f"seg_{index:04d}"
    return result


class Diarizer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def assign_speakers(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        if self.settings.engine_mode == "demo":
            return segments
        try:
            return self._pyannote_split(audio_path, segments)
        except Exception as exc:
            logger.warning("pyannote diarization unavailable, using ECAPA clustering: %s", exc)
        try:
            return self._speechbrain_cluster(audio_path, segments)
        except Exception as exc:
            # Diarization is explicitly best-effort. Keeping the transcript usable is
            # more valuable than failing the whole job when embeddings are unavailable.
            # The pause heuristic only runs here: it must not override an embedding
            # decision that both turns are the same voice.
            logger.warning("Speaker embeddings unavailable, using pause heuristic: %s", exc)
            return self._short_dialogue_fallback(segments)

    def _pyannote_split(self, audio_path: Path, segments: list[Segment]) -> list[Segment]:
        import soundfile
        import torch
        from pyannote.audio import Pipeline

        token = self.settings.huggingface_hub_token or None
        try:
            pipeline = Pipeline.from_pretrained(PYANNOTE_PIPELINE, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(PYANNOTE_PIPELINE, use_auth_token=token)
        if pipeline is None:
            raise RuntimeError("pyannote pipeline could not be loaded (gated model terms not accepted?)")
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        samples, sample_rate = soundfile.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
        output = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=self.settings.num_speakers,
        )
        annotation = getattr(output, "speaker_diarization", output)
        turns = [
            (float(turn.start), float(turn.end), str(label))
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
        if not turns:
            raise RuntimeError("pyannote found no speaker turns")
        return split_segments_by_turns(segments, turns)

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
            if end - start < sample_rate * MIN_EMBED_SECONDS:
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
            inherit_short_segment_speakers(segments, usable_indices)
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
        inherit_short_segment_speakers(segments, usable_indices)
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
