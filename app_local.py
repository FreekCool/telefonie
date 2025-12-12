import os
import json
import time
import base64
import re
import asyncio
from pathlib import Path
import wave
import audioop
import array
import webrtcvad

from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse

# -------------------------------------------------
# env / config
# -------------------------------------------------

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PUBLIC_BASE = os.getenv("PUBLIC_BASE_URL")      # e.g. https://<your-ngrok>.ngrok-free.app
WHISPER_API_URL = os.getenv("WHISPER_API_URL")  # e.g. https://xxxx-8000.proxy.runpod.net

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise ValueError("Missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN")
if not PUBLIC_BASE:
    raise ValueError("Missing PUBLIC_BASE_URL (e.g. https://<your-tunnel>)")
if not WHISPER_API_URL:
    raise ValueError("Missing WHISPER_API_URL (e.g. https://xxxx-8000.proxy.runpod.net)")

PORT = int(os.getenv("PORT", 5055))

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# OpenMP hack (harmless even if Whisper now runs remotely)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------------------------------
# local TTS + Mistral + remote STT
# -------------------------------------------------

# --- STT (remote Whisper) + TTS + Mistral ---

from remote_stt_client import RemoteSpeechToText
from local_stt_tts import TextToSpeech   # we only use TTS from this file
from mistral_client import ask_mistral, chat_mistral, chat_mistral_async, chat_mistral_stream_async

WHISPER_API_URL = os.getenv("WHISPER_API_URL")
if not WHISPER_API_URL:
    raise ValueError("Missing WHISPER_API_URL in .env")

stt = RemoteSpeechToText(
    base_url=WHISPER_API_URL,
    language="nl",
)

tts = TextToSpeech(
    piper_path="piper",
    model_path="piper_models/nl_BE-nathalie-medium.onnx",
)

# -------------------------------------------------
# system prompt + greeting from files
# -------------------------------------------------

def load_text_file(filepath: str, default: str = "") -> str:
    """
    Load text from a file, with fallback to default if file doesn't exist.
    """
    try:
        path = Path(filepath)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        else:
            print(f"Warning: {filepath} not found, using default.")
            return default
    except Exception as e:
        print(f"Error loading {filepath}: {e}, using default.")
        return default


SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
SYSTEM_PROMPT_DEFAULT = (
    "Je bent een vriendelijke Nederlandse telefonische assistent. "
    "Antwoord in 1 of 2 korte zinnen, duidelijk en beleefd. "
    "Geen opsommingen, geen lijstjes."
)
SYSTEM_PROMPT = load_text_file(SYSTEM_PROMPT_PATH, SYSTEM_PROMPT_DEFAULT)

GREETING_PATH = os.getenv("GREETING_PATH", "greeting.txt")
GREETING_DEFAULT = (
    "Hallo, je spreekt met jouw telefonische assistent. "
    "Wat kan ik voor je doen?"
)
GREETING = load_text_file(GREETING_PATH, GREETING_DEFAULT)

# --- VAD / turn-taking constants (tuned to be a bit snappier) ---

CHUNK_MS = 20                 # Twilio sends 20ms audio chunks
MIN_BUFFER_BYTES = 4000       # minimum bytes (~0.5s) before we bother sending to STT (reduced for faster response)
MAX_UTTER_BYTES = 40000       # hard cap: if user talks long, cut and process anyway

app = FastAPI()

# -------------------------------------------------
# helpers
# -------------------------------------------------


def clean_for_tts(text: str) -> str:
    """
    Maak tekst natuurlijker voor TTS met betere prosodie en natuurlijke pauzes.
    """
    if not text:
        return ""
    
    # Remove markdown and formatting
    text = re.sub(r"^[\-\*\d\.\)]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Remove bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Remove italic
    
    # Expand common abbreviations for more natural speech
    abbreviations = {
        r'\bdr\.': 'dokter',
        r'\bmr\.': 'meneer',
        r'\bmvr\.': 'mevrouw',
        r'\betc\.': 'enzovoort',
        r'\bbv\.': 'bijvoorbeeld',
        r'\bi\.v\.m\.': 'in verband met',
        r'\bo\.a\.': 'onder andere',
        r'\bz\.o\.z\.': 'zo snel mogelijk',
    }
    for abbrev, expansion in abbreviations.items():
        text = re.sub(abbrev, expansion, text, flags=re.IGNORECASE)
    
    # Convert numbers to words for more natural speech (simple cases)
    def number_to_words(match):
        num = int(match.group())
        if num < 20:
            numbers = ['nul', 'een', 'twee', 'drie', 'vier', 'vijf', 'zes', 'zeven', 
                      'acht', 'negen', 'tien', 'elf', 'twaalf', 'dertien', 'veertien',
                      'vijftien', 'zestien', 'zeventien', 'achttien', 'negentien']
            return numbers[num] if num < len(numbers) else match.group()
        return match.group()  # Keep larger numbers as digits
    
    # Only convert single-digit numbers in certain contexts
    text = re.sub(r'\b([0-9]{1,2})\b', number_to_words, text)
    
    # Add natural pauses: replace commas with comma + slight pause
    # Replace sentence endings with longer pause
    text = re.sub(r'([.!?])\s+', r'\1 ', text)  # Normalize sentence endings
    text = re.sub(r',\s+', ', ', text)  # Normalize commas
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Ensure proper sentence capitalization
    sentences = re.split(r'([.!?]\s+)', text)
    result = []
    for i, part in enumerate(sentences):
        if part and part[0].isalpha():
            part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
        result.append(part)
    text = ''.join(result)
    
    # Limit length
    if len(text) > 400:
        # Try to cut at sentence boundary
        cut_point = text[:400].rfind('.')
        if cut_point > 200:  # Only cut at sentence if reasonable
            text = text[:cut_point + 1]
        else:
            text = text[:400]
    
    return text


def resample_pcm16(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono PCM16 using audioop."""
    if src_rate == dst_rate:
        return pcm_bytes
    sample_width = 2
    nchannels = 1
    converted, _ = audioop.ratecv(
        pcm_bytes,
        sample_width,
        nchannels,
        src_rate,
        dst_rate,
        None,
    )
    return converted


def ulaw_to_wav(mu_bytes: bytes, path: str, rate: int = 8000, channels: int = 1):
    """Save μ-law bytes as PCM16 WAV."""
    pcm16 = audioop.ulaw2lin(mu_bytes, 2)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16)


def save_stereo(in_mu: bytes, out_mu: bytes, path: str, rate: int = 8000):
    """Interleave inbound/outbound as stereo WAV (L=inbound, R=outbound)."""
    pcmL = audioop.ulaw2lin(in_mu, 2)
    pcmR = audioop.ulaw2lin(out_mu, 2)

    L = array.array("h")
    L.frombytes(pcmL)
    R = array.array("h")
    R.frombytes(pcmR)

    m = max(len(L), len(R))
    if len(L) < m:
        L.extend([0] * (m - len(L)))
    if len(R) < m:
        R.extend([0] * (m - len(R)))

    interleaved = array.array("h", (s for pair in zip(L, R) for s in pair))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(interleaved.tobytes())


async def send_tts_to_twilio(
    text: str,
    websocket: WebSocket,
    stream_sid: str,
    out_ulaw: bytearray,
    stop_event: asyncio.Event,
):
    """TTS → PCM → 8kHz → μ-law → Twilio media events (in chunks)."""
    if not text or not text.strip():
        return  # Skip empty text to avoid Piper errors
    
    reply_text = clean_for_tts(text)
    if not reply_text or not reply_text.strip():
        return  # Skip if cleaned text is empty
    
    # Use optimized parameters for natural, conversational speech
    # length_scale=0.95: slightly faster for natural pace
    # noise_scale=0.7: more prosody variation for expressiveness  
    # noise_w=0.85: more rhythm variation for natural flow
    reply_pcm, sr = tts.synthesize_to_bytes(
        reply_text,
        length_scale=0.95,  # Slightly faster for natural conversation pace
        noise_scale=0.7,     # More variation for expressive prosody
        noise_w=0.85         # More rhythm variation for natural flow
    )
    reply_pcm_8k = resample_pcm16(reply_pcm, sr, 8000)
    ulaw_reply = audioop.lin2ulaw(reply_pcm_8k, 2)

    # optioneel: opslaan voor later
    out_ulaw.extend(ulaw_reply)

    # stuur in kleine chunks en pace op real-time om barge-in effectief te maken
    chunk_size = 80  # bytes μ-law ≈ 10ms bij 8kHz → minimale backlog voor barge-in
    seconds_per_chunk = chunk_size / 8000.0  # 10ms pacing
    for i in range(0, len(ulaw_reply), chunk_size):
        if stop_event.is_set():
            break
        chunk = ulaw_reply[i: i + chunk_size]
        payload = base64.b64encode(chunk).decode("utf-8")

        try:
            # Check if websocket is still open before sending
            if websocket.client_state.name != "CONNECTED":
                break
            await websocket.send_json(
                {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }
            )
            # pace sending so we don't flood Twilio; allows clear to be effective
            await asyncio.sleep(seconds_per_chunk)
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            # Websocket closed, stop sending
            break


async def stream_mistral_to_tts(
    conversation: list,
    websocket: WebSocket,
    stream_sid: str,
    out_ulaw: bytearray,
    stop_event: asyncio.Event,
    temperature: float = 0.25,
    timeout: int = 8,
    max_tokens: int = 100,
) -> str:
    """
    Stream Mistral response and feed chunks to TTS with proper buffering.
    Guards against TTS getting ahead by only sending complete sentences/phrases.
    Returns the full response text for conversation history.
    """
    import re
    
    full_response = ""
    buffer = ""
    MIN_SENTENCE_LENGTH = 15  # minimum chars before sending a sentence (guards against TTS getting ahead)
    MIN_PHRASE_LENGTH = 25    # minimum chars for comma-separated phrases
    
    try:
        # Use async streaming - non-blocking
        async for chunk in chat_mistral_stream_async(
            conversation, 
            model="mistral-small-latest",
            temperature=temperature, 
            timeout=timeout, 
            max_tokens=max_tokens
        ):
            if stop_event.is_set():
                break
            
            full_response += chunk
            buffer += chunk
            
            # Only send to TTS when we have enough buffered content
            # This guards against TTS getting ahead of the stream
            
            # Check for sentence endings first (highest priority)
            sentence_match = re.search(r'[.!?]\s+', buffer)
            if sentence_match and len(buffer[:sentence_match.end()]) >= MIN_SENTENCE_LENGTH:
                chunk_to_speak = buffer[:sentence_match.end()].strip()
                buffer = buffer[sentence_match.end():]
                
                if chunk_to_speak and not stop_event.is_set():
                    await send_tts_to_twilio(chunk_to_speak, websocket, stream_sid, out_ulaw, stop_event)
                continue
            
            # Check for comma pauses (only if we have enough text)
            comma_match = re.search(r',\s+', buffer)
            if comma_match and len(buffer[:comma_match.end()]) >= MIN_PHRASE_LENGTH:
                chunk_to_speak = buffer[:comma_match.end()].strip()
                buffer = buffer[comma_match.end():]
                
                if chunk_to_speak and not stop_event.is_set():
                    await send_tts_to_twilio(chunk_to_speak, websocket, stream_sid, out_ulaw, stop_event)
                continue
            
            # If buffer is getting very long, send it anyway (safety valve)
            if len(buffer) >= 80:
                chunk_to_speak = buffer.strip()
                buffer = ""
                if chunk_to_speak and not stop_event.is_set():
                    await send_tts_to_twilio(chunk_to_speak, websocket, stream_sid, out_ulaw, stop_event)
        
        # Send any remaining buffer (only if meaningful and complete)
        if buffer and buffer.strip() and len(buffer.strip()) >= MIN_SENTENCE_LENGTH and not stop_event.is_set():
            await send_tts_to_twilio(buffer.strip(), websocket, stream_sid, out_ulaw, stop_event)
            
    except Exception as e:
        print(f"Streaming error: {e}")
        # Fallback: send what we have if meaningful
        if full_response.strip() and len(full_response.strip()) >= MIN_SENTENCE_LENGTH and not stop_event.is_set():
            await send_tts_to_twilio(full_response.strip(), websocket, stream_sid, out_ulaw, stop_event)
    
    return full_response


def start_recording_when_ready(call_sid: str):
    for attempt in range(12):
        try:
            call = client.calls(call_sid).fetch()
            print(f"[recording] call {call_sid} status={call.status}")
            if call.status == "in-progress":
                try:
                    rec = client.calls(call_sid).recordings.create(
                        recording_channels="dual",
                        recording_track="both",
                        recording_status_callback=f"{PUBLIC_BASE}/recording-callback",
                        recording_status_callback_event=["in-progress", "completed", "failed"],
                    )
                    print(f"✅ Recording started for {call_sid} (RecordingSid={rec.sid})")
                    return
                except Exception as e:
                    print("❌ start recording in-progress failed:", e)
                    try:
                        rec = client.calls(call_sid).recordings.create(
                            recording_channels="mono",
                            recording_track="both",
                            recording_status_callback=f"{PUBLIC_BASE}/recording-callback",
                            recording_status_callback_event=["in-progress", "completed", "failed"],
                        )
                        print(f"✅ Recording started (mono) for {call_sid} (RecordingSid={rec.sid})")
                        return
                    except Exception as e2:
                        print("❌ mono fallback failed:", e2)
                        return
            elif call.status in {"completed", "failed", "busy", "no-answer", "canceled"}:
                print(f"[recording] call ended/not eligible (status={call.status})")
                return
        except Exception as e:
            print("❌ fetch call status failed:", e)
        time.sleep(0.8)
    print("⚠️ gave up waiting for in-progress; no recording started")


# -------------------------------------------------
# HTTP routes
# -------------------------------------------------


@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server (remote Whisper / local Mistral/TTS) is running!"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request, background_tasks: BackgroundTasks):
    response = VoiceResponse()

    form = await request.form()
    call_sid = form.get("CallSid")
    print("Incoming CallSid:", call_sid)

    if call_sid and PUBLIC_BASE:
        background_tasks.add_task(start_recording_when_ready, call_sid)

    host = request.url.hostname
    connect = Connect()
    stream_el = connect.stream(url=f"wss://{host}/media-stream")
    if call_sid:
        stream_el.parameter(name="callSid", value=call_sid)

    response.append(connect)
    return HTMLResponse(str(response), media_type="application/xml")


@app.post("/recording-callback")
async def rec_cb(request: Request):
    form = await request.form()
    data = dict(form)
    sid = data.get("RecordingSid")
    call_sid = data.get("CallSid")
    status = data.get("RecordingStatus")
    rec_url = data.get("RecordingUrl")
    if rec_url and not rec_url.startswith("http"):
        rec_url = f"https://api.twilio.com{rec_url}"
    mp3_url = f"{rec_url}.mp3" if rec_url else None

    print("🎧 Recording callback:", data)
    if mp3_url:
        print(f"🔗 Recording MP3: {mp3_url}")
        print("   (use Basic Auth with your TWILIO_ACCOUNT_SID as username and TWILIO_AUTH_TOKEN)")
        print(f'   curl -u "{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}" -o {sid}.mp3 "{mp3_url}"')

    return HTMLResponse("OK")


# -------------------------------------------------
# WebSocket: Twilio Media Stream
# -------------------------------------------------


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """
    Simpele turn-based voice bot:
    - buffer inbound audio
    - bij stilte -> STT -> Mistral -> TTS
    - stuur antwoord terug als één blok (in chunks) audio
    """
    await websocket.accept()
    print("Client connected (remote Whisper pipeline)")

    in_ulaw = bytearray()
    out_ulaw = bytearray()
    stream_sid = None
    call_sid_from_param = None
    # conversation history (system prompt + rolling messages)
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    # reservation state (extracted from conversation for faster context)
    reservation_state = {
        "aantal_personen": None,
        "datum": None,
        "tijd": None,
        "naam": None,
    }
    # playback control
    stop_playback = asyncio.Event()
    tts_task = None
    speaking = False

    # === WebRTC VAD parameters (optimized for low latency) ===
    vad = webrtcvad.Vad(2)        # Mode 2: balanced sensitivity for faster detection
    VAD_SPEECH_FRAMES = 5         # need N consecutive speech frames (~100ms) to start (faster)
    VAD_SILENCE_FRAMES = 15       # need N consecutive silence frames (~300ms) to end (faster)
    MIN_UTTER_VOICE_MS = 300      # require at least this much voiced audio (reduced for faster response)
    MAX_UTTERANCE_MS = 5000       # safety cut-off (5s) to keep turns short
    MIN_RMS_FOR_UTTER = 250       # drop utterances whose overall RMS is below this (bg noise)

    speech_frames = 0
    silence_frames = 0
    in_speech = False
    utter_ms = 0
    voiced_ms_total = 0
    processing = False
    barge_in_speech_frames = 0  # track consecutive speech frames for barge-in

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid_from_param = data["start"].get("customParameters", {}).get("callSid")
                print(f"Stream started {stream_sid} CallSid={call_sid_from_param}")
                
                # Reset conversation and reservation state for new call
                conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
                reservation_state = {
                    "aantal_personen": None,
                    "datum": None,
                    "tijd": None,
                    "naam": None,
                }

                greeting = GREETING
                if tts_task and not tts_task.done():
                    stop_playback.set()
                    tts_task.cancel()
                stop_playback.clear()

                async def run_tts(text):
                    nonlocal speaking
                    speaking = True
                    try:
                        await send_tts_to_twilio(text, websocket, stream_sid, out_ulaw, stop_playback)
                    except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError, ConnectionError):
                        # Normal cancellation or websocket closed, ignore
                        pass
                    finally:
                        speaking = False

                tts_task = asyncio.create_task(run_tts(greeting))

            elif event == "media":
                payload_b64 = data["media"]["payload"]
                ulaw_chunk = base64.b64decode(payload_b64)
                in_ulaw.extend(ulaw_chunk)

                # μ-law -> PCM16 voor VAD (8 kHz mono, 20ms frame)
                pcm_chunk = audioop.ulaw2lin(ulaw_chunk, 2)

                # WebRTC VAD expects 10/20/30ms frames; Twilio sends 20ms.
                is_speech = vad.is_speech(pcm_chunk, sample_rate=8000)
                rms = audioop.rms(pcm_chunk, 2)

                # barge-in: require 3-4 consecutive speech frames (~60-80ms) to avoid false triggers
                if speaking:
                    if is_speech and rms > 500:  # both VAD and RMS must indicate speech
                        barge_in_speech_frames += 1
                        if barge_in_speech_frames >= 3:  # ~60ms of clear speech
                            print("Barge-in: stopping playback due to user speech.")
                            stop_playback.set()
                            if stream_sid:
                                await websocket.send_json({"event": "clear", "streamSid": stream_sid})
                            if tts_task and not tts_task.done():
                                tts_task.cancel()
                            speaking = False
                            barge_in_speech_frames = 0
                    else:
                        barge_in_speech_frames = 0  # reset counter on silence/noise
                else:
                    barge_in_speech_frames = 0  # reset when not speaking

                if is_speech:
                    speech_frames += 1
                    silence_frames = 0
                    voiced_ms_total += CHUNK_MS
                    if speech_frames >= VAD_SPEECH_FRAMES:
                        in_speech = True
                else:
                    speech_frames = 0
                    if in_speech:
                        silence_frames += 1

                if in_speech:
                    utter_ms += CHUNK_MS
                else:
                    # don't let utter_ms grow during pure silence; prevents idle max_length triggers
                    utter_ms = 0

                # criteria om te knippen:
                end_by_silence = in_speech and silence_frames >= VAD_SILENCE_FRAMES
                end_by_length = in_speech and utter_ms >= MAX_UTTERANCE_MS

                if (
                    (end_by_silence or end_by_length)
                    and len(in_ulaw) >= MIN_BUFFER_BYTES
                    and not processing
                ):
                    processing = True
                    reason = "silence" if end_by_silence else "max_length"
                    print(f"Detected end of utterance ({reason}), sending to Whisper...")

                    pcm_all = audioop.ulaw2lin(bytes(in_ulaw), 2)
                    in_ulaw.clear()
                    speech_frames = 0
                    silence_frames = 0
                    in_speech = False
                    utter_ms = 0
                    voiced_ms_snapshot = voiced_ms_total
                    voiced_ms_total = 0
                    barge_in_speech_frames = 0  # reset barge-in counter

                    # ignore if too little voiced audio (likely background noise)
                    if voiced_ms_snapshot < MIN_UTTER_VOICE_MS:
                        print("Utterance skipped: not enough speech detected.")
                        processing = False
                        continue

                    # ignore if overall RMS is tiny (steady background noise)
                    overall_rms = audioop.rms(pcm_all, 2) if pcm_all else 0
                    if overall_rms < MIN_RMS_FOR_UTTER:
                        print(f"Utterance skipped: RMS too low ({overall_rms}).")
                        processing = False
                        continue

                    # --- STT via Runpod Whisper (async for non-blocking) ---
                    # Provide context to improve accuracy for restaurant/reservation vocabulary
                    whisper_prompt = (
                        "Restaurant telefoongesprek. Belangrijke woorden: reservering, reserveren, "
                        "bestelling, menu, pizzeria, tafel, personen, datum, tijd, naam, adres, "
                        "bezorgen, ophalen, allergie, lactose."
                    )
                    user_text = await stt.transcribe_pcm_async(
                        pcm_all, 
                        sample_rate=8000,
                        initial_prompt=whisper_prompt
                    )
                    print("User said:", repr(user_text))

                    if not user_text.strip():
                        # Lege transcript: negeren, geen "Sorry..." spam
                        print("Empty transcript from Whisper, skipping reply.")
                        processing = False
                        continue

                    # --- LLM (Mistral) with smart history compression ---
                    conversation.append({"role": "user", "content": user_text})
                    
                    # Extract reservation information from user message
                    # Extract number of people (improved pattern matching)
                    personen_match = re.search(r'(\d+)\s*(?:personen?|mensen?|gasten?)', user_text.lower())
                    if personen_match:
                        reservation_state["aantal_personen"] = personen_match.group(1)
                    else:
                        # Check for written numbers
                        number_words = {
                            'vier': '4', 'tien': '10', 'twaalf': '12', 'acht': '8',
                            'twee': '2', 'drie': '3', 'vijf': '5', 'zes': '6',
                            'zeven': '7', 'negen': '9', 'elf': '11'
                        }
                        for word, num in number_words.items():
                            if word in user_text.lower():
                                reservation_state["aantal_personen"] = num
                                break
                    
                    # Extract date
                    if any(word in user_text.lower() for word in ['morgen', '14 december', 'december']):
                        if 'morgen' in user_text.lower():
                            reservation_state["datum"] = "morgen"
                        elif '14 december' in user_text.lower() or 'december' in user_text.lower():
                            reservation_state["datum"] = "14 december"
                    
                    # Extract time
                    tijd_match = re.search(r'(\d+)\s*uur|half\s*(\d+)|(\d+):(\d+)', user_text.lower())
                    if tijd_match:
                        if 'half' in user_text.lower():
                            reservation_state["tijd"] = f"half {tijd_match.group(2) or tijd_match.group(1)}"
                        elif ':' in user_text:
                            reservation_state["tijd"] = tijd_match.group(0)
                        else:
                            reservation_state["tijd"] = f"{tijd_match.group(1)} uur"
                    
                    # Extract name
                    naam_patterns = [
                        r'(?:ik ben|mijn naam is|dit is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                        r'([A-Z][a-z]+\s+van\s+[A-Z][a-z]+)',  # "Bart van Vliet"
                    ]
                    for pattern in naam_patterns:
                        naam_match = re.search(pattern, user_text)
                        if naam_match:
                            reservation_state["naam"] = naam_match.group(1)
                            break
                    
                    # Smart compression: keep only recent messages (2-3 exchanges) but add reservation state
                    if len(conversation) > 1 + 6:  # Keep only last 3 exchanges (6 messages)
                        # Build context summary from reservation state
                        context_parts = []
                        if reservation_state["aantal_personen"]:
                            context_parts.append(f"aantal personen: {reservation_state['aantal_personen']}")
                        if reservation_state["datum"]:
                            context_parts.append(f"datum: {reservation_state['datum']}")
                        if reservation_state["tijd"]:
                            context_parts.append(f"tijd: {reservation_state['tijd']}")
                        if reservation_state["naam"]:
                            context_parts.append(f"naam: {reservation_state['naam']}")
                        
                        # Update system prompt with reservation context if we have any
                        if context_parts:
                            context_str = "Belangrijke informatie uit het gesprek: " + ", ".join(context_parts) + "."
                            enhanced_system = SYSTEM_PROMPT + "\n\n" + context_str
                            conversation = [{"role": "system", "content": enhanced_system}] + conversation[-6:]
                        else:
                            conversation = [conversation[0]] + conversation[-6:]

                    # --- TTS -> Twilio (background, barge-in aware, streaming) ---
                    if tts_task and not tts_task.done():
                        stop_playback.set()
                        tts_task.cancel()
                    stop_playback.clear()

                    async def run_streaming_tts():
                        nonlocal speaking
                        speaking = True
                        try:
                            # Stream Mistral response and feed to TTS with proper buffering
                            reply_raw = await stream_mistral_to_tts(
                                conversation,
                                websocket,
                                stream_sid,
                                out_ulaw,
                                stop_playback,
                                temperature=0.25,
                                timeout=8,
                                max_tokens=100,
                            )
                            # Save full response to conversation history
                            conversation.append({"role": "assistant", "content": reply_raw})
                            print("Mistral reply (raw):", reply_raw)
                        except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError, ConnectionError):
                            # Normal cancellation or websocket closed, ignore
                            pass
                        except Exception as e:
                            # keep the call alive on upstream hiccups
                            print("LLM streaming failed, using fallback reply:", e)
                            fallback = "Sorry, ik had even geen verbinding. Waarmee kan ik u helpen?"
                            conversation.append({"role": "assistant", "content": fallback})
                            if not stop_playback.is_set():
                                await send_tts_to_twilio(fallback, websocket, stream_sid, out_ulaw, stop_playback)
                        finally:
                            speaking = False

                    tts_task = asyncio.create_task(run_streaming_tts())

                    processing = False

            elif event == "stop":
                print("Stream stopped", stream_sid)
                break

    except WebSocketDisconnect:
        print("Client disconnected from media-stream.")
    except Exception as e:
        print("Error in handle_media_stream:", e)
    finally:
        # lokaal opslaan van inbound/outbound audio voor debug
        try:
            os.makedirs("recordings", exist_ok=True)
            base = call_sid_from_param or stream_sid or str(int(time.time()))
            in_path = f"recordings/{base}_inbound.wav"
            out_path = f"recordings/{base}_outbound.wav"
            st_path = f"recordings/{base}_stereo.wav"

            if in_ulaw:
                ulaw_to_wav(in_ulaw, in_path)
            if out_ulaw:
                ulaw_to_wav(out_ulaw, out_path)
            if in_ulaw or out_ulaw:
                save_stereo(in_ulaw, out_ulaw, st_path)

            print(
                f"🎧 Saved: {in_path if in_ulaw else '(no inbound)'} "
                f"{out_path if out_ulaw else '(no outbound)'} "
                f"{st_path}"
            )
        except Exception as e:
            print("Failed to write WAVs:", e)

        # Twilio recordings fallback printen
        try:
            if call_sid_from_param:
                recs = client.recordings.list(call_sid=call_sid_from_param, limit=5)
                if not recs:
                    print(f"⚠️ No Twilio recordings found for CallSid {call_sid_from_param}")
                for r in recs:
                    api_url = r.media_url if r.media_url.startswith("http") else f"https://api.twilio.com{r.media_url}"
                    mp3 = f"{api_url}.mp3"
                    print(f"🔗 Twilio recording: SID={r.sid}")
                    print(f"   {mp3}")
                    print(
                        f'   curl -u "{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}" '
                        f'-o {r.sid}.mp3 "{mp3}"'
                    )
        except Exception as e:
            print("Failed to fetch Twilio recordings:", e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)