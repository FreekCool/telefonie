import time

in_wav = 'input_question.wav'

t0 = time.time()
user_text = stt.transcribe_file(in_wav)
print("STT duurde:", time.time() - t0, "seconden")