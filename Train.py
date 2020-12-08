import Utils
import numpy as np
import Preprocessor
import sklearn
import os


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
    files = Utils.scan_file(config["data"]["train_dataset_dir"])
    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epoch"]
    name_list = config["data"]["names"]
    nb_classes = config["model"]["nb_classes"]
    classifier = select_classifier(
        config["model"]["model_name"],
        config["model"]["input_shape"],
        config["model"]["nb_classes"],
        config["model"]["model_path"])
    if nb_classes != len(name_list):
        print("配置文件出错，分类目标数目与名单不符")
        exit(-1)
    y_train = []
    x_train = []
    for f in files:
        name = os.path.splitext(f)[0]
        index = -1
        try:
            index = name_list.index(name)
        except:
            pass
        if index != -1:
            print("加载"+f+"......")
            y, sr = preprocessor.LoadFile(
                config["data"]["train_dataset_dir"]+f)
            onsets = preprocessor.ConvertOnsetCut(y, sr)
            ans = []
            for i in range(len(onsets)):
                ans.append(index)
            x_train = x_train + list(onsets)
            y_train = y_train + list(ans)
        else:
            print("无法识别的文件名："+f)

    y_train = np.array(y_train, dtype='int')
    tmp = np.array(range(nb_classes))
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(tmp.reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    x_train = np.array(x_train)

    classifier.fit(x_train, y_train, batch_size, epochs)
