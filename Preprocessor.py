import librosa
import librosa.display
import numpy as np


class preprocessor():
    def __init__(self, mode="test", length=8):
        self.mode = mode
        self.length = length

    def LoadFile(self, file_path):
        y, _ = librosa.load(file_path)
        return y

    def ConvertOnsetCut(self, y):
        onsets_frames = librosa.onset.onset_detect(y)
        results = []
        D = librosa.stft(y)
        # D = librosa.amplitude_to_db(np.abs(D))
        for i in onsets_frames:
            results.append(np.array(D[:, i]).reshape((1025, 1, 1)))
        return results


if __name__ == "__main__":
    preprocessor = preprocessor()
    y = preprocessor.LoadFile(r'data/PIANO/p0.wav')
    resutls = preprocessor.ConvertOnsetCut(y)
    print(resutls)
