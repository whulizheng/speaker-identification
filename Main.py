import Utils
import numpy as np
import Preprocessor
import sklearn


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


if __name__ == "__main__":
    config = Utils.readjson("config.json")
    preprocessor = Preprocessor.preprocessor()
    data_names = ["p0.wav", "p1.wav", "p2.wav", "p3.wav", "p4.wav", "p5.wav"]
    ans_names = ["p0.csv", "p1.csv", "p2.csv", "p3.csv", "p4.csv", "p5.csv"]
    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epoch"]
    y_train = []
    x_train = []
    for i in range(len(data_names)):
        y = preprocessor.LoadFile("data\\train\\" + data_names[i])
        ans = Utils.Readans("data\\train\\" + ans_names[i])
        onsets = preprocessor.ConvertOnsetCut(y)
        if len(ans) != len(onsets):
            print("音源不清晰，"+data_names[i]+"被跳过，请更换音源，谢谢合作")
            continue
        x_train = x_train + list(onsets)
        y_train = y_train + list(ans)
    y_train = np.array(y_train, dtype='int')
    tmp = np.array(range(88))
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(tmp.reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    x_train = np.array(x_train)
    classifier = select_classifier(
        config["model"]["model_name"],
        config["model"]["input_shape"],
        config["model"]["nb_classes"],
        config["model"]["model_path"])
    classifier.fit(x_train, y_train, batch_size, epochs)
    # 下面是测试部分
    classifier.load_model("model/best_model.hdf5")
    y = preprocessor.LoadFile("data/test/2.wav")  # ans = 8 6 4 9 4 8 3 5 0
    onsets = preprocessor.ConvertOnsetCut(y)
    for on in onsets:
        y_pred = classifier.predict(np.array(on).reshape((1, 1025, 1, 1)))
        print(y_pred)
