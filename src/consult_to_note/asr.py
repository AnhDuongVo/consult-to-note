"""Speech recognition with NVIDIA Parakeet through the Riva gRPC API.

Works against the hosted endpoint on build.nvidia.com (grpc.nvcf.nvidia.com:443 + function-id) or a
self-hosted Parakeet NIM (localhost:50051, no SSL). Input: 16-bit PCM WAV (mono, 16 kHz works best).

Two ways to get speakers:
* `transcribe_two_channel`: one recording per speaker (as in PriMock57). Exact speaker labels.
* `transcribe_file`: one mixed recording. Uses Riva speaker diarization if the model supports it,
  otherwise the pipeline labels speakers with the fast LLM (`label_speakers`).
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from .config import Settings
from .llm import LLM, complete_structured
from .prompts import LABEL_SPEAKERS
from .schemas import Speaker, Transcript, Utterance


@dataclass
class Word:
    text: str
    start_s: float
    end_s: float
    speaker_tag: int | None = None


def _service(settings: Settings):
    try:
        import riva.client
    except ImportError as err:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the ASR extra: pip install -e '.[asr]'") from err
    metadata = []
    if settings.asr_function_id:
        metadata.append(["function-id", settings.asr_function_id])
    if settings.api_key and settings.asr_use_ssl:  # never send the key over a plaintext connection
        metadata.append(["authorization", f"Bearer {settings.api_key}"])
    auth = riva.client.Auth(uri=settings.asr_server, use_ssl=settings.asr_use_ssl, metadata_args=metadata)
    return riva.client, riva.client.ASRService(auth)


def _wav_chunks(path: Path, max_seconds: float) -> list[tuple[float, bytes]]:
    """Split a WAV file into chunks (offset seconds, wav bytes) small enough for offline requests."""
    chunks = []
    with wave.open(str(path), "rb") as wf:
        rate, width, channels = wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
        if width != 2:
            raise ValueError(f"{path}: expected 16-bit PCM WAV, got {8 * width}-bit")
        frames_per_chunk = int(rate * max_seconds)
        offset = 0.0
        while True:
            frames = wf.readframes(frames_per_chunk)
            if not frames:
                break
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setnchannels(channels)
                out.setsampwidth(width)
                out.setframerate(rate)
                out.writeframes(frames)
            chunks.append((offset, buf.getvalue()))
            offset += len(frames) / (width * channels * rate)
    return chunks


def recognize_words(
    path: str | Path,
    settings: Settings,
    *,
    diarization: bool = False,
    max_speakers: int = 2,
    boost_words: list[str] | None = None,
    boost_score: float = 20.0,
    chunk_seconds: float | None = 25.0,
) -> list[Word]:
    """Offline recognition with word timestamps (and speaker tags when diarization is available).

    `chunk_seconds=None` sends the whole file in one request. Diarization needs that: speaker tags are
    only consistent within one request.
    """
    riva, asr = _service(settings)
    config = riva.RecognitionConfig(
        language_code=settings.asr_language,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        verbatim_transcripts=False,
    )
    if boost_words:
        # Word boosting helps with drug names and rare clinical terms.
        riva.add_word_boosting_to_config(config, boost_words, boost_score)
    if diarization:
        riva.add_speaker_diarization_to_config(config, True, max_speakers)

    words: list[Word] = []
    chunks = _wav_chunks(Path(path), chunk_seconds) if chunk_seconds else [(0.0, Path(path).read_bytes())]
    for offset, data in chunks:
        resp = asr.offline_recognize(data, config)
        for result in resp.results:
            if not result.alternatives:
                continue
            for w in result.alternatives[0].words:
                tag = getattr(w, "speaker_tag", None) if diarization else None
                # Riva reports word times in milliseconds.
                words.append(Word(w.word, offset + w.start_time / 1000, offset + w.end_time / 1000, tag))
    return words


def words_to_utterances(words: list[tuple[Speaker, Word]], gap_s: float = 1.2) -> list[Utterance]:
    """Group time-ordered words into utterances at speaker changes or long pauses."""
    utterances: list[Utterance] = []
    current: list[Word] = []
    current_speaker: Speaker | None = None
    for speaker, w in sorted(words, key=lambda sw: sw[1].start_s):
        new_turn = current_speaker is not None and (
            speaker != current_speaker or (current and w.start_s - current[-1].end_s > gap_s)
        )
        if new_turn:
            utterances.append(
                Utterance(
                    id=len(utterances) + 1,
                    speaker=current_speaker,
                    text=" ".join(x.text for x in current),
                    start_s=current[0].start_s,
                    end_s=current[-1].end_s,
                )
            )
            current = []
        current.append(w)
        current_speaker = speaker
    if current and current_speaker is not None:
        utterances.append(
            Utterance(
                id=len(utterances) + 1,
                speaker=current_speaker,
                text=" ".join(x.text for x in current),
                start_s=current[0].start_s,
                end_s=current[-1].end_s,
            )
        )
    return utterances


def transcribe_two_channel(
    doctor_wav: str | Path, patient_wav: str | Path, settings: Settings, boost_words: list[str] | None = None
) -> Transcript:
    """Transcribe one file per speaker and interleave by time. Exact speaker attribution."""
    tagged: list[tuple[Speaker, Word]] = []
    for speaker, path in (("doctor", doctor_wav), ("patient", patient_wav)):
        tagged += [(speaker, w) for w in recognize_words(path, settings, boost_words=boost_words)]
    return Transcript(utterances=words_to_utterances(tagged), source=f"asr:{Path(doctor_wav).name}")


class _Labels(BaseModel):
    labels: list[Speaker]


async def label_speakers(transcript: Transcript, llm: LLM) -> Transcript:
    """Assign doctor/patient to segments whose speaker is unknown, using the fast model."""
    segments = "\n".join(f"{i + 1}. {u.text}" for i, u in enumerate(transcript.utterances))
    result = await complete_structured(
        llm,
        [
            {
                "role": "user",
                "content": LABEL_SPEAKERS.format(segments=segments)
                + f"\nReturn exactly {len(transcript.utterances)} labels in order.",
            }
        ],
        _Labels,
        role="fast",
        step="label_speakers",
    )
    labels = (result.labels + ["other"] * len(transcript.utterances))[: len(transcript.utterances)]
    for u, label in zip(transcript.utterances, labels, strict=True):
        u.speaker = label
    return transcript


async def transcribe_file(
    path: str | Path,
    settings: Settings,
    llm: LLM | None = None,
    diarization: bool = True,
    boost_words: list[str] | None = None,
) -> Transcript:
    """Transcribe one mixed recording. Speakers come from diarization, or from the LLM as a fallback.

    Note: the LLM fallback sends the (unmasked) transcript to the configured model endpoint.
    """
    words: list[Word] = []
    if diarization:
        try:
            # One request for the whole file, so speaker tags stay consistent across the recording.
            words = recognize_words(path, settings, diarization=True, boost_words=boost_words, chunk_seconds=None)
        except Exception:
            words = []  # model or file size does not allow it: fall back to chunks + LLM labelling
    if not words:
        words = recognize_words(path, settings, diarization=False, boost_words=boost_words)
    has_tags = any(w.speaker_tag not in (None, 0) for w in words) or len({w.speaker_tag for w in words}) > 1
    if has_tags:
        tags = sorted({w.speaker_tag for w in words})
        # Assumption: whoever speaks first is the doctor; the LLM can correct this downstream.
        first = words[0].speaker_tag if words else None
        mapping = {t: ("doctor" if t == first else "patient") for t in tags}
        utterances = words_to_utterances([(mapping[w.speaker_tag], w) for w in words])
        return Transcript(utterances=utterances, source=f"asr:{Path(path).name}")
    utterances = words_to_utterances([("other", w) for w in words])
    transcript = Transcript(utterances=utterances, source=f"asr:{Path(path).name}")
    if llm is not None:
        transcript = await label_speakers(transcript, llm)
    return transcript
