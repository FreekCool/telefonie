# test_tts.py
from local_stt_tts import TextToSpeech

def main():
    tts = TextToSpeech(
        piper_path="piper",  # now it's on PATH thanks to pip
        model_path="piper_models/nl_BE-nathalie-medium.onnx"
    )

    text = "Hallo, hoe gaat het met je?"
    out_wav = "input_question.wav"

    print("Generating speech...")
    tts.synthesize_to_wav(text, out_wav)
    print(f"Klaar! Bestaat nu: {out_wav}")

if __name__ == "__main__":
    main()