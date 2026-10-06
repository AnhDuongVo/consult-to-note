"""Speech event sources for the live scribe.

* `simulate_from_transcript`: replays a text transcript at speaking pace. No audio or ASR needed, so you
  can measure the LLM side of the latency budget on its own.
* `stream_wav`: Parakeet streaming ASR through Riva, sending a WAV file at real-time pace.
* `stream_two_channel`: one stream per speaker (PriMock57 layout), merged into one event stream.
* `stream_microphone`: live microphone input (needs `pip install pyaudio`).

All sources yield `SpeechEvent`s; only `is_final` events trigger note updates.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..config import Settings
from ..schemas import Transcript


@dataclass
class SpeechEvent:
    speaker: str
    text: str
    is_final: bool
    audio_end_s: float | None  # position in the recording
    wall_time: float  # time.perf_counter() when the event was emitted (start of the latency clock)


async def simulate_from_transcript(
    transcript: Transcript, words_per_second: float = 2.6, speed: float = 1.0
) -> AsyncIterator[SpeechEvent]:
    clock = 0.0
    for u in transcript.utterances:
        duration = max(0.4, len(u.text.split()) / words_per_second)
        await asyncio.sleep(duration / speed)
        clock += duration
        yield SpeechEvent(u.speaker, u.text, True, clock, time.perf_counter())


def _riva_streaming(settings: Settings):
    try:
        import riva.client
    except ImportError as err:  # pragma: no cover
        raise RuntimeError("pip install -e '.[asr]' for streaming ASR") from err
    metadata = []
    if settings.asr_function_id:
        metadata.append(["function-id", settings.asr_function_id])
    if settings.api_key:
        metadata.append(["authorization", f"Bearer {settings.api_key}"])
    auth = riva.client.Auth(uri=settings.asr_server, use_ssl=settings.asr_use_ssl, metadata_args=metadata)
    asr = riva.client.ASRService(auth)
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            language_code=settings.asr_language,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            verbatim_transcripts=False,
        ),
        interim_results=True,
    )
    return riva.client, asr, config


async def _from_thread(produce, *args) -> AsyncIterator[SpeechEvent]:
    """Run a blocking gRPC generator in a thread and hand events to asyncio."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def worker():
        try:
            for ev in produce(*args):
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        except Exception as err:  # surface errors in the event loop
            loop.call_soon_threadsafe(queue.put_nowait, err)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await queue.get()
        if item is done:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def _wav_events(path: str, settings: Settings, speaker: str, realtime: bool, chunk_frames: int):
    riva, asr, config = _riva_streaming(settings)
    delay = riva.sleep_audio_length if realtime else None
    with riva.AudioChunkFileIterator(path, chunk_frames, delay) as chunks:
        for resp in asr.streaming_response_generator(audio_chunks=chunks, streaming_config=config):
            for r in resp.results:
                if not r.alternatives:
                    continue
                yield SpeechEvent(
                    speaker,
                    r.alternatives[0].transcript.strip(),
                    bool(r.is_final),
                    getattr(r, "audio_processed", None),
                    time.perf_counter(),
                )


async def stream_wav(
    path: str, settings: Settings, speaker: str = "other", realtime: bool = True, chunk_frames: int = 1600
) -> AsyncIterator[SpeechEvent]:
    async for ev in _from_thread(_wav_events, path, settings, speaker, realtime, chunk_frames):
        yield ev


async def stream_two_channel(doctor_wav: str, patient_wav: str, settings: Settings) -> AsyncIterator[SpeechEvent]:
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    async def pump(path: str, speaker: str):
        try:
            async for ev in stream_wav(path, settings, speaker):
                await queue.put(ev)
        finally:
            await queue.put(sentinel)

    tasks = [asyncio.create_task(pump(doctor_wav, "doctor")), asyncio.create_task(pump(patient_wav, "patient"))]
    finished = 0
    while finished < len(tasks):
        item = await queue.get()
        if item is sentinel:
            finished += 1
            continue
        yield item
    for t in tasks:
        t.result()  # re-raise errors


def _mic_events(settings: Settings, device: int | None):
    _riva, asr, config = _riva_streaming(settings)
    from riva.client.audio_io import MicrophoneStream

    rate = 16000
    config.config.sample_rate_hertz = rate
    config.config.audio_channel_count = 1
    with MicrophoneStream(rate, 1600, device=device) as stream:
        for resp in asr.streaming_response_generator(audio_chunks=stream, streaming_config=config):
            for r in resp.results:
                if r.alternatives:
                    yield SpeechEvent(
                        "other", r.alternatives[0].transcript.strip(), bool(r.is_final), None, time.perf_counter()
                    )


async def stream_microphone(settings: Settings, device: int | None = None) -> AsyncIterator[SpeechEvent]:
    async for ev in _from_thread(_mic_events, settings, device):
        yield ev
