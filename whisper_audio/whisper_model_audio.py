import whisper
import glob
import time

path = 'nagrania_testowe'

if __name__ == '__main__':
    start = time.time()
    #Load model
    model = whisper.load_model("base")

    for file in glob.glob(path + "/*.wav"):
        print(file)
        audio = whisper.load_audio(file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

    end = time.time()
    print(end - start)
