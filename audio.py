from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
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
    duration = 0
    for file in glob.glob(path_to_prepare + "/*.mp3"):
        current_signal, _ = librosa.load(file, sr=16000)
        current_signal = librosa.to_mono(current_signal)
        #current_signal = current_signal[:int(len(current_signal) / 8)] #Cut signal
        duration = duration + librosa.get_duration(current_signal)
        name = path + '/' + os.path.basename(file)[:-4] + '.wav'
        sf.write(name, current_signal, samplerate=16000)
    print(duration)


def load_signal(path):
    """""
    Load sounds
    """""
    nagrania = []
    nagrania_nazwy =[]

    for file in glob.glob(path + "/*.wav"):
        current_signal, _ = read_wave(file)
        nagrania.append(current_signal)
        nagrania_nazwy.append(os.path.basename(file)[:-4])
    return nagrania, nagrania_nazwy


def check_prediction(prediction, k_max):
    """""
    Check prediction
    """""
    if prediction == 'Polish':
        global k_true
        k_true += 1
    Accuracy = k_true/k_max * 100
    print("Accuracy: {}%".format(Accuracy))


class JSON:
    def __init__(self, st, en, lang, lang_conf):
        self.start = st
        self.end = en
        self.language = lang
        self.language_conf = lang_conf

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    def update_json(self, path):
        data = json.load(open(path))
        if type(data) is dict:
            data = [data]
        data.append(self)
        return json.dumps(data, default=lambda o: o.__dict__, indent=4)


if __name__ == '__main__':
    #start = time.time()


    #Load Sounds
    #prepare_signal('nagrania')
    start = time.time()
    counter = 0

    sounds, sounds_name = load_signal(path)

    #Load model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-lang-id")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("anton-l/wav2vec2-base-lang-id", ignore_mismatched_sizes=True)
    print(model.config.id2label) #possible languages

    #VAD MODE
    for i in range(len(sounds)):
        index_start = 0
        for j, seg in enumerate(get_segments(sounds[i], sample_rate=16000, minimal_silence_dur=150, minimal_segment_duration=30, maximal_segment_duration=30)):
            st, en, dur, segment = seg
            segment_file = path_output + '/vad/' + sounds_name[i] + '_{} '.format(j) + '.wav'
            write_wave(segment_file, segment, 16000)

            #Prepare input
            input_signal, _ = librosa.load(segment_file, sr=16000)
            inputs = feature_extractor(input_signal, return_tensors="pt", sampling_rate=16000)
            
            #Detect Language
            with torch.no_grad():
                logits = model(**inputs).logits

            predicted_class_ids = torch.argmax(logits, dim=-1).item()
            predicted_label = model.config.id2label[predicted_class_ids]
            counter += 1

            # Language confidence
            softmax = torch.nn.Softmax(dim=-1)
            language_conf = softmax(logits)
            language_conf = language_conf.tolist()

            print("Sound: {}_{}".format(sounds_name[i], j))
            print("Time: {}-{}".format(index_start, round(index_start + float(dur), 2)))
            print("Prediction: {}".format(predicted_label))
            print("Language confidence: {}%".format(round(language_conf[0][predicted_class_ids]*100, 2)))
            check_prediction(predicted_label, counter)
            print("----------------")

            #JSON
            json_object = JSON(index_start, round(index_start + float(dur), 2), predicted_label, round
            (language_conf[0][predicted_class_ids] * 100, 2))
            if j == 0:
                json_object = json_object.to_json()
            else:
                json_object = json_object.update_json(path_output + '/' + sounds_name[i] + '.json')
            with open(path_output + '/' + sounds_name[i] + '.json', 'w') as jfile:
                jfile.write(json_object)
            index_start = round(index_start + float(dur), 2)

    end = time.time()
    print(end - start)
