import librosa
import json
import numpy as np


def LoadFile(file_path):
    y, _ = librosa.load(file_path)
    return y


def Readans(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    return data


def readjson(address):
    with open(address, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m" % '   '] *
                  int(percent * total_length)) + ''
    bar = '\r' + start_str + \
        bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)
