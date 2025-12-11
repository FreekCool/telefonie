# remote_stt_client.py
import io
import wave
import audioop
from typing import Optional

import requests
import httpx
import asyncio


class RemoteSpeechToText:
    """
    Simple client for the Whisper HTTP API running on Runpod.
    Sends WAV over HTTP, gets back plain text.
    """

    def __init__(self, base_url: str, language: str = "nl"):
        self.base_url = base_url.rstrip("/")
        self.language = language

    def _pcm16_to_wav_bytes(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        """
        Wrap raw PCM16 mono bytes into a WAV container in-memory.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int = 8000) -> str:
        """
        Convert PCM16 mono -> 16 kHz WAV -> POST to /transcribe.
        No initial_prompt, just raw Whisper output.
        """
        if not pcm_bytes:
            return ""

        # Resample 8 kHz -> 16 kHz for Whisper
        if sample_rate != 16000:
            pcm_bytes, _ = audioop.ratecv(
                pcm_bytes,
                2,          # sample width
                1,          # mono
                sample_rate,
                16000,
                None,
            )
            sample_rate = 16000

        wav_bytes = self._pcm16_to_wav_bytes(pcm_bytes, sample_rate)

        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
        }
        data = {
            "language": self.language,
        }

        url = f"{self.base_url}/transcribe"
        resp = requests.post(url, files=files, data=data, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        return (j.get("text") or "").strip()
    
    async def transcribe_pcm_async(self, pcm_bytes: bytes, sample_rate: int = 8000) -> str:
        """
        Async version: non-blocking HTTP call for faster processing.
        Convert PCM16 mono -> 16 kHz WAV -> POST to /transcribe.
        """
        if not pcm_bytes:
            return ""

        # Resample 8 kHz -> 16 kHz for Whisper
        if sample_rate != 16000:
            pcm_bytes, _ = audioop.ratecv(
                pcm_bytes,
                2,          # sample width
                1,          # mono
                sample_rate,
                16000,
                None,
            )
            sample_rate = 16000

        wav_bytes = self._pcm16_to_wav_bytes(pcm_bytes, sample_rate)

        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
        }
        data = {
            "language": self.language,
        }

        url = f"{self.base_url}/transcribe"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, files=files, data=data)
            resp.raise_for_status()
            j = resp.json()
            return (j.get("text") or "").strip()