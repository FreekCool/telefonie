# local_stt_tts.py
from pathlib import Path
import subprocess
import tempfile
from typing import Tuple

# Handle audioop deprecation in Python 3.13+
try:
    import audioop
except ImportError:
    try:
        import pyaudioop as audioop
    except ImportError:
        raise ImportError(
            "audioop module not found. Please install pyaudioop: pip install pyaudioop\n"
            "Or use Python 3.11/3.12 where audioop is built-in."
        )

import soundfile as sf
from faster_whisper import WhisperModel
import numpy as np


# local_stt_tts.py
class SpeechToText:
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        language: str = "nl",
        initial_prompt: str | None = None,
    ):
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type="int8"
        )
        self.language = language
        self.initial_prompt = initial_prompt

    def transcribe_file(self, wav_path: str) -> str:
        segments, info = self.model.transcribe(
            wav_path,
            language=self.language,
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )
        text = "".join(seg.text for seg in segments)
        return text.strip()

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int = 8000) -> str:
        if not pcm_bytes:
            return ""

        # 8k -> 16k
        if sample_rate != 16000:
            pcm_bytes, _ = audioop.ratecv(
                pcm_bytes,
                2,
                1,
                sample_rate,
                16000,
                None,
            )
            sample_rate = 16000

        audio_array = np.frombuffer(pcm_bytes, dtype="int16").astype("float32") / 32768.0

        segments, info = self.model.transcribe(
            audio_array,
            language=self.language,
            beam_size=5,          # iets hoger dan 3 -> betere kwaliteit
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )
        text = "".join(seg.text for seg in segments)
        return text.strip()


class TextToSpeech:
    def __init__(self, piper_path: str, model_path: str):
        """
        piper_path: e.g. 'piper' (CLI, installed via pip)
        model_path: path to .onnx model, e.g. 'piper_models/nl_NL-mls-medium.onnx'
        """
        self.piper_path = piper_path
        self.model_path = model_path

    def synthesize_to_wav(self, text: str, out_wav_path: str, length_scale: float = 1.0, noise_scale: float = 0.667, noise_w: float = 0.8):
        """
        Synthesize text to WAV with natural prosody parameters.
        
        Args:
            text: Text to synthesize
            out_wav_path: Output WAV file path
            length_scale: Speech rate (lower = faster, higher = slower). Default 1.0, try 0.9-1.1 for variation
            noise_scale: Variation in prosody (0.5-0.8). Higher = more variation, more natural
            noise_w: Phoneme duration variation (0.6-1.0). Higher = more natural rhythm
        """
        out_path = Path(out_wav_path)

        cmd = [
            self.piper_path,
            "--model", self.model_path,
            "--output_file", str(out_path),
            "--length_scale", str(length_scale),
            "--noise_scale", str(noise_scale),
            "--noise_w", str(noise_w),
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(text)

        if proc.returncode != 0:
            raise RuntimeError(f"Piper failed with code {proc.returncode}:\n{stderr}")

    def synthesize_to_bytes(self, text: str, length_scale: float = 0.95, noise_scale: float = 0.7, noise_w: float = 0.85) -> Tuple[bytes, int]:
        """
        Return (pcm16_bytes, sample_rate) for in-memory use.
        
        Uses optimized parameters for more natural, conversational speech:
        - Slightly faster (0.95) for more natural pace
        - Higher variation (0.7) for more expressive prosody
        - More rhythm variation (0.85) for natural flow
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        self.synthesize_to_wav(text, tmp_path, length_scale=length_scale, noise_scale=noise_scale, noise_w=noise_w)

        data, sr = sf.read(tmp_path, dtype="int16")
        Path(tmp_path).unlink(missing_ok=True)

        if data.ndim > 1:
            data = data.mean(axis=1).astype("int16")

        pcm_bytes = data.tobytes()
        return pcm_bytes, sr