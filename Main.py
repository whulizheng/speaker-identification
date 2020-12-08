import Utils
import librosa
from tqdm import tqdm
import pandas
import numpy as np
import Preprocessor
from moviepy.editor import VideoFileClip

target_path = "C:\\Users\\WhuLi\\Documents\\Tmp\\圆桌派第四季20190829.mp4"
names = ["窦文涛", "梁文道", "马家辉", "周轶君", "背景音乐"]


def select_classifier(classifier_name, input_shape, nb_classes, output_directory):
    if classifier_name == "fcn":
        import Classifier_FCN
        return Classifier_FCN.Classifier_FCN(
            input_shape=input_shape,
            nb_classes=nb_classes,
            output_directory=output_directory
        )
    else:
        print("不支持的分类器")
        exit(-1)


config = Utils.readjson("config.json")
clip = VideoFileClip(target_path)
t_duration = clip.duration
audio = clip.audio
audio.write_audiofile('tmp\\audio.wav')
preprocessor = Preprocessor.preprocessor()
classifier = select_classifier(
    config["model"]["model_name"],
    config["model"]["input_shape"],
    config["model"]["nb_classes"],
    config["model"]["model_path"])
y, sr = preprocessor.LoadFile('tmp\\audio.wav')
D = librosa.stft(y)
lenth = len(D[0])
onsets_frames = librosa.onset.onset_detect(y, sr, backtrack=True)
onsets = preprocessor.ConvertOnsetCut(y, sr)
classifier.load_model('model\\best_model.hdf5')
time_line = []
speakers = []
for s in tqdm(onsets_frames):
    time_line.append((int(s)/lenth)*t_duration)
for s in tqdm(onsets):
    ans = config["data"]["names"][int(classifier.predict(
        np.array(s).reshape((1, 1025*3, 1, 1))))]
    speakers.append(ans)
df = pandas.DataFrame()
df[0] = time_line
df[1] = speakers
df.to_csv('tmp\\pd_data.csv', header=None, index=None)
print("finished")
