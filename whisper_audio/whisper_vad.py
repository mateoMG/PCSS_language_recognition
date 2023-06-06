import whisper
import torch
import glob
import os
import librosa
import soundfile as sf
import json
import time

from audiosplit import read_wave, write_wave, get_segments

path = 'nagrania_testowe'
path_output = 'output'
k_true = 0


def prepare_signal(path_to_prepare):
    """""
    Tools to prepare sounds
    """""
    for file in glob.glob(path_to_prepare + "/*.mp3"):
        current_signal, _ = librosa.load(file, sr=16000)
        current_signal = librosa.to_mono(current_signal)
        # current_signal = current_signal[:int(len(current_signal) / 8)] #Cut signal
        name = path + '/' + os.path.basename(file)[:-4] + '.wav'
        sf.write(name, current_signal, samplerate=16000)


def load_signal(path):
    """""
    Load sounds
    """""
    nagrania = []
    nagrania_nazwy = []

    for file in glob.glob(path + "/*.wav"):
        current_signal, _ = read_wave(file)
        nagrania.append(current_signal)
        nagrania_nazwy.append(os.path.basename(file)[:-4])
    return nagrania, nagrania_nazwy


def check_prediction(prediction, k_max):
    """""
    Check prediction
    """""
    if prediction == 'pl':
        global k_true
        k_true += 1
    Accuracy = k_true / k_max * 100
    print("Accuracy: {}%".format(Accuracy))


class JSON:
    def __init__(self, st, en, lang):
        self.start = st
        self.end = en
        self.language = lang

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    def update_json(self, path):
        data = json.load(open(path))
        if type(data) is dict:
            data = [data]
        data.append(self)
        return json.dumps(data, default=lambda o: o.__dict__, indent=4)


if __name__ == '__main__':


    # Load Sounds
    # prepare_signal('nagrania')
    start = time.time()
    counter = 0
    sounds, sounds_name = load_signal(path)

    # Load model
    model = whisper.load_model("base")

    # VAD MODE
    for i in range(len(sounds)):
        index_start = 0
        for j, seg in enumerate(
                get_segments(sounds[i], sample_rate=16000, minimal_silence_dur=150, minimal_segment_duration=10,
                             maximal_segment_duration=10)):
            st, en, dur, segment = seg
            segment_file = path_output + '/vad/' + sounds_name[i] + '_{} '.format(j) + '.wav'
            write_wave(segment_file, segment, 16000)

            # Prepare input
            audio = whisper.load_audio(segment_file)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)


            # Detect Language
            _, probs = model.detect_language(mel)
            counter += 1



            print("Sound: {}_{}".format(sounds_name[i], j))
            print("Time: {}-{}".format(index_start, round(index_start + float(dur), 2)))
            print(f"Detected language: {max(probs, key=probs.get)}")
            check_prediction(max(probs, key=probs.get), counter)
            print("----------------")

            # JSON
            json_object = JSON(index_start, round(index_start + float(dur), 2), max(probs, key=probs.get))
            if j == 0:
                json_object = json_object.to_json()
            else:
                json_object = json_object.update_json(path_output + '/' + sounds_name[i] + '.json')
            with open(path_output + '/' + sounds_name[i] + '.json', 'w') as jfile:
                jfile.write(json_object)
            index_start = round(index_start + float(dur), 2)

    end = time.time()
    print(end - start)
