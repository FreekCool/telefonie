from flask import Flask, Response
from twilio.twiml.voice_response import VoiceResponse
# from flask_wtf.csrf import CSRFProtect
# csrf = CSRFProtect(app)

app = Flask(__name__)

# @csrf.exempt
@app.route("/voice", methods=["GET","POST"])
def voice():
    resp = VoiceResponse()
    resp.say("Met snackbar de kromme kroket, hoe kan ik je helpen? Testbericht over", voice="alice", language="nl-NL")
    return Response(str(resp), mimetype="text/xml")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=False)