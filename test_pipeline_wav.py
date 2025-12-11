# test_pipeline_wav.py

import os
import time
import re
import requests

from dotenv import load_dotenv

# Zorg dat env-variabelen geladen worden (.env)
load_dotenv()

# URL van je Whisper API op Runpod, bv:
# WHISPER_API_URL="https://<pod>-8000.proxy.runpod.net"
WHISPER_API_URL = os.getenv("WHISPER_API_URL")

if not WHISPER_API_URL:
    raise RuntimeError("WHISPER_API_URL ontbreekt in .env")

# eventueel: strip trailing slash
WHISPER_API_URL = WHISPER_API_URL.rstrip("/")

# OpenMP-fix is nu eigenlijk niet meer nodig (geen lokale faster-whisper),
# maar kan geen kwaad om te laten staan.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from local_stt_tts import TextToSpeech        # alleen TTS lokaal
from mistral_client import ask_mistral


def clean_for_tts(text: str) -> str:
    """
    Maak de Mistral-output iets simpeler voor de TTS:
    - bullets/nummering weg
    - whitespace opschonen
    - lengte limiteren
    """
    text = re.sub(r"^[\-\*\d\.\)]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if len(text) > 400:
        text = text[:400]
    return text


def transcribe_via_whisper_api(wav_path: str, language: str = "nl") -> str:
    """
    Stuurt een WAV-bestand naar de Whisper API op Runpod
    en geeft de herkende tekst terug.
    """
    url = f"{WHISPER_API_URL}/transcribe"

    with open(wav_path, "rb") as f:
        files = {
            "file": (os.path.basename(wav_path), f, "audio/wav"),
        }
        data = {
            "language": language,
        }
        resp = requests.post(url, files=files, data=data, timeout=120)

    resp.raise_for_status()
    payload = resp.json()
    return payload.get("text", "").strip()


def main():
    tts = TextToSpeech(
        piper_path="piper",
        model_path="piper_models/nl_BE-nathalie-medium.onnx",
    )

    in_wav = "input_question.wav"
    out_wav = "mistral_reply.wav"

    # 1) STT via Runpod Whisper
    print(f"Transcribing {in_wav} via Whisper API ...")
    t0 = time.time()
    user_text = transcribe_via_whisper_api(in_wav, language="nl")
    dt = time.time() - t0
    print("STT duurde:", round(dt, 3), "seconden")
    print("User said:", user_text)

    # 2) Mistral
    system_prompt = (
        "Je bent een vriendelijke Nederlandse telefonische assistent van snackbar De Balken. "
        "Antwoord in 1 of 2 korte zinnen, duidelijk en beleefd. "
        "Geen opsommingen, geen lijstjes, geen markdown."
    )
    reply_text = ask_mistral(user_text, system_prompt=system_prompt)
    print("Mistral reply (raw):", reply_text)

    # 3) Schoonmaken voor TTS
    reply_text = clean_for_tts(reply_text)
    print("Cleaned for TTS:", reply_text)

    # 4) TTS
    print(f"Generating TTS to {out_wav} ...")
    tts.synthesize_to_wav(reply_text, out_wav)
    print("Done! Luister naar:", out_wav)


if __name__ == "__main__":
    main()