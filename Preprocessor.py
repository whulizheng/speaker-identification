import librosa
import librosa.display
import numpy as np


class preprocessor():
    def __init__(self, mode="test", length=8):
        self.mode = mode
        self.length = length

    def LoadFile(self, file_path):
        y, sr = librosa.load(file_path)
        return y, sr

    def ConvertOnsetCut(self, y, sr):
        onsets_frames = librosa.onset.onset_detect(y, sr=sr, backtrack=True)
        results = []
        D = librosa.stft(y)
        # D = librosa.amplitude_to_db(np.abs(D))
        for i in onsets_frames:
            results.append(
                np.array([D[:, i], D[:, i+1], D[:, i+2]]).reshape((1025*3, 1, 1)))
        return results


if __name__ == "__main__":
    preprocessor = preprocessor()
    y, sr = preprocessor.LoadFile(r'data/PIANO/p0.wav')
    resutls = preprocessor.ConvertOnsetCut(y, sr)
    print(resutls)
