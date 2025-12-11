import os
import json
import time
import base64
import asyncio
import websockets
from pathlib import Path
import wave, audioop, array
from twilio.rest import Client
from fastapi import BackgroundTasks
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PUBLIC_BASE = os.getenv("PUBLIC_BASE_URL")  # e.g. https://<your-tunnel>
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise ValueError("Missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN")
if not PUBLIC_BASE:
    raise ValueError("Missing PUBLIC_BASE_URL (e.g. https://<your-tunnel>)")
if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

PORT = int(os.getenv('PORT', 5055))
DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
    "Always stay positive, but work in a joke when appropriate."
)
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]
SHOW_TIMING_MATH = False

app = FastAPI()

SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
_cache = {"mtime": None, "text": DEFAULT_SYSTEM_MESSAGE}

def get_system_message() -> str:
    p = Path(SYSTEM_PROMPT_PATH)
    try:
        mtime = p.stat().st_mtime
        if _cache["mtime"] != mtime:
            _cache["text"] = p.read_text(encoding="utf-8")
            _cache["mtime"] = mtime
    except FileNotFoundError:
        # keep default if file missing
        pass
    return _cache["text"]

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
                        # 👇 get callbacks when it starts and when it finishes (and on failure)
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
            elif call.status in {"completed","failed","busy","no-answer","canceled"}:
                print(f"[recording] call ended/not eligible (status={call.status})")
                return
        except Exception as e:
            print("❌ fetch call status failed:", e)
        time.sleep(0.8)
    print("⚠️ gave up waiting for in-progress; no recording started")

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request, background_tasks: BackgroundTasks):
    response = VoiceResponse()

    # (no response.say here to avoid robotic greeting)

    # Get CallSid from Twilio's webhook POST
    form = await request.form()
    call_sid = form.get("CallSid")
    print("Incoming CallSid:", call_sid)

    # ✅ Start recording AFTER Twilio begins executing this TwiML
    if call_sid and PUBLIC_BASE:
        background_tasks.add_task(start_recording_when_ready, call_sid)

    # Connect Twilio <Stream> to your WS and pass CallSid as a <Parameter>
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
    # Twilio posts x-www-form-urlencoded fields like:
    # RecordingSid, RecordingUrl (no extension), RecordingStatus, CallSid, etc.
    data = dict(form)
    sid = data.get("RecordingSid")
    call_sid = data.get("CallSid")
    status = data.get("RecordingStatus")
    # RecordingUrl is path-only or full API URL without extension
    rec_url = data.get("RecordingUrl")  # e.g. /2010-04-01/Accounts/AC.../Recordings/RE...
    # Build a direct mp3 URL that works with basic auth (AC SID + Auth Token)
    if rec_url and not rec_url.startswith("http"):
        rec_url = f"https://api.twilio.com{rec_url}"
    mp3_url = f"{rec_url}.mp3" if rec_url else None

    print("🎧 Recording callback:", data)
    if mp3_url:
        print(f"🔗 Recording MP3: {mp3_url}")
        print("   (use Basic Auth with your TWILIO_ACCOUNT_SID as username and TWILIO_AUTH_TOKEN as password)")
        print(f'   curl -u "{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}" -o {sid}.mp3 "{mp3_url}"')

    return HTMLResponse("OK")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("Client connected")
    await websocket.accept()

    # --- Buffers for local recording (μ-law, 8kHz) ---
    in_ulaw  = bytearray()   # caller → you
    out_ulaw = bytearray()   # you → caller
    call_sid_from_param = None
    stream_sid = None
    latest_media_timestamp = 0
    last_assistant_item = None
    mark_queue = []
    response_start_timestamp_twilio = None

    def ulaw_to_wav(mu_bytes: bytes, path: str, rate: int = 8000, channels: int = 1):
        """Save μ-law bytes as PCM16 WAV."""
        pcm16 = audioop.ulaw2lin(mu_bytes, 2)  # 2 bytes/sample (16-bit)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm16)

    def save_stereo(in_mu: bytes, out_mu: bytes, path: str, rate: int = 8000):
        """Interleave inbound/outbound as stereo WAV (L=inbound, R=outbound)."""
        pcmL = audioop.ulaw2lin(in_mu,  2)
        pcmR = audioop.ulaw2lin(out_mu, 2)
        L = array.array('h'); L.frombytes(pcmL)
        R = array.array('h'); R.frombytes(pcmR)
        m = max(len(L), len(R))
        if len(L) < m: L.extend([0]*(m - len(L)))
        if len(R) < m: R.extend([0]*(m - len(R)))
        interleaved = array.array('h', (s for pair in zip(L, R) for s in pair))
        with wave.open(path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(interleaved.tobytes())

    async with websockets.connect(
        "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview",
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await initialize_session(openai_ws)

        async def ai_greet():
            await openai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": (
                            "Begroet de beller in natuurlijk, vriendelijk Nederlands namens cafetaria De Balken. "
                            "Klink menselijk en kort, introduceer jezelf als de telefonische AI assisten van cafetaria De Balken en vraag vriendelijk eerst of de klant wil een bestelling wil laten bezorgen of ophalen."
                        )
                    }]
                }
            }))
            await openai_ws.send(json.dumps({"type": "response.create"}))

        # --- helpers for barge-in & Twilio mark/clear flow ---
        async def send_mark(connection, stream_sid_local):
            if stream_sid_local:
                await connection.send_json({
                    "event": "mark",
                    "streamSid": stream_sid_local,
                    "mark": {"name": "responsePart"}
                })
                mark_queue.append("responsePart")

        async def handle_speech_started_event():
            nonlocal response_start_timestamp_twilio, last_assistant_item, latest_media_timestamp, stream_sid
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed = max(0, latest_media_timestamp - response_start_timestamp_twilio)
                if last_assistant_item:
                    await openai_ws.send(json.dumps({
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed
                    }))
                if stream_sid:
                    await websocket.send_json({
                        "event": "clear",
                        "streamSid": stream_sid
                    })
                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def receive_from_twilio():
            nonlocal stream_sid, latest_media_timestamp, call_sid_from_param
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)

                    if data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        call_sid_from_param = (data['start'].get('customParameters', {}).get('callSid'))
                        print(f"Incoming stream has started {stream_sid} callSid={call_sid_from_param}")

                        # <-- have the AI speak first (natural greeting)
                        await ai_greet()

                    elif data['event'] == 'media' and openai_ws.open:
                        latest_media_timestamp = int(data['media']['timestamp'])
                        # buffer inbound audio (caller → you)
                        in_ulaw.extend(base64.b64decode(data['media']['payload']))

                        # forward to OpenAI
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }))

                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)

            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)

                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        # buffer outbound audio (you → caller)
                        chunk = base64.b64decode(response['delta'])
                        out_ulaw.extend(chunk)

                        # send to Twilio
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": { "payload": base64.b64encode(chunk).decode("utf-8") }
                        }
                        await websocket.send_json(audio_delta)

                        if response_start_timestamp_twilio is None:
                            response_start_timestamp_twilio = latest_media_timestamp

                        if response.get('item_id'):
                            last_assistant_item = response['item_id']

                        await send_mark(websocket, stream_sid)

                    if response.get('type') == 'input_audio_buffer.speech_started':
                        if last_assistant_item:
                            await handle_speech_started_event()

            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        try:
            await asyncio.gather(receive_from_twilio(), send_to_twilio())
        finally:
            # Save the recordings when the socket closes
            os.makedirs("recordings", exist_ok=True)
            base = call_sid_from_param or stream_sid or str(int(time.time()))
            in_path  = f"recordings/{base}_inbound.wav"
            out_path = f"recordings/{base}_outbound.wav"
            st_path  = f"recordings/{base}_stereo.wav"

            try:
                if in_ulaw:
                    ulaw_to_wav(in_ulaw, in_path)
                if out_ulaw:
                    ulaw_to_wav(out_ulaw, out_path)
                if in_ulaw or out_ulaw:
                    save_stereo(in_ulaw, out_ulaw, st_path)
                print(f"🎧 Saved: {in_path if in_ulaw else '(no inbound)'} "
                      f"{out_path if out_ulaw else '(no outbound)'} "
                      f"{st_path}")
            except Exception as e:
                print("Failed to write WAVs:", e)

            # --- Fallback: list Twilio recordings for this call and print links ---
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
                        print(f'   curl -u "{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}" -o {r.sid}.mp3 "{mp3}"')
            except Exception as e:
                print("Failed to fetch Twilio recordings:", e)

async def send_initial_conversation_item(openai_ws):
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there! I am an AI voice assistant powered by Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or anything you can imagine. How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))

async def initialize_session(openai_ws):
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            # NEW: force Dutch speech recognition
            "input_audio_transcription": {
                "model": "gpt-4o-transcribe",
                "language": "nl"  # or "nl-NL"
            },
            # Slightly slower, more natural pacing
            "speed": 1,
            # Make the speaking style super explicit
            "instructions": get_system_message(),
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)