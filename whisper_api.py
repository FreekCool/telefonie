from typing import Optional
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# -----------------------------
# Model init (runs at startup)
# -----------------------------

MODEL_NAME = "large-v3"

print(f"[Whisper API] Loading model '{MODEL_NAME}' on CUDA (A4500)...")
model = WhisperModel(
    MODEL_NAME,
    device="cuda",              # use GPU
    compute_type="int8_float16" # safe on 20GB, fast & high quality
)
print("[Whisper API] Model loaded.")

app = FastAPI(title="Whisper STT API")

# -----------------------------
# Helpers
# -----------------------------

def run_transcribe(path: str, language: str = "nl", initial_prompt: Optional[str] = None) -> str:
    """
    Run faster-whisper on a file and return the combined text.
    Adjust beam_size etc. here if you want to trade speed/quality.
    """
    # Default initial prompt with restaurant/reservation vocabulary for better accuracy
    default_prompt = (
        "Dit is een telefoongesprek met een restaurant. "
        "Woorden die vaak voorkomen: reservering, reserveren, bestelling, menu, "
        "pizzeria, restaurant, tafel, personen, datum, tijd, naam, adres, "
        "bezorgen, ophalen, allergie, lactose, vegetarisch."
    )
    
    prompt = initial_prompt or default_prompt
    
    segments, info = model.transcribe(
        path,
        language=language,
        beam_size=5,                    # higher = better, slower
        best_of=5,                      # try multiple candidates
        temperature=0.0,                # deterministic, more accurate
        vad_filter=False,               # we already do VAD client-side
        condition_on_previous_text=True, # use context from previous segments
        initial_prompt=prompt,          # context for better accuracy
        word_timestamps=False,          # faster without word timestamps
    )
    text = "".join(seg.text for seg in segments)
    return text.strip()

# -----------------------------
# Routes
# -----------------------------

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Whisper STT API running",
        "model": MODEL_NAME,
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
):
    """
    Accepts an audio file (WAV/MP3/OGG/etc.) via multipart/form-data as 'file'
    and returns {"text": "..."}.
    
    Optional form fields:
    - language: Language code (default: "nl")
    - initial_prompt: Context prompt for better accuracy
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(contents)
            tmp.flush()
            text = run_transcribe(
                tmp.name, 
                language=language or "nl", 
                initial_prompt=initial_prompt
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    return JSONResponse({"text": text})