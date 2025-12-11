from typing import Optional
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# -----------------------------
# Model init (runs at startup)
# -----------------------------

# You can change "large-v3" to "medium" or "small" if you want.
MODEL_NAME = "large-v3"

print(f"[Whisper API] Loading model '{MODEL_NAME}' on CUDA...")
model = WhisperModel(
    MODEL_NAME,
    device="cuda",          # use GPU
    compute_type="float16", # good for L4 / most modern GPUs
)
print("[Whisper API] Model loaded.")


app = FastAPI(title="Whisper STT API")


# -----------------------------
# Helpers
# -----------------------------

def run_transcribe(path: str, language: str = "nl") -> str:
    """
    Run faster-whisper on a file and return the combined text.
    Adjust beam_size etc. here if you want to trade speed/quality.
    """
    segments, info = model.transcribe(
        path,
        language=language,
        beam_size=5,                    # higher = better, slower
        vad_filter=False,               # we already chunk on the client side
        condition_on_previous_text=False,
    )
    text = "".join(seg.text for seg in segments)
    return text.strip()


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
async def root():
    return {"status": "ok", "message": "Whisper STT API running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = "nl",
):
    """
    Accepts an audio file (WAV/MP3/OGG/etc.) via multipart/form-data as 'file'
    and returns {"text": "..."}.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(contents)
            tmp.flush()
            text = run_transcribe(tmp.name, language=language or "nl")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    return JSONResponse({"text": text})