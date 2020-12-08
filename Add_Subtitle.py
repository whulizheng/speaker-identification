import Utils
import moviepy
import numpy as np
import pandas
import gc
from tqdm import tqdm
from moviepy import editor

subtitle_file_path = "tmp\\pd_data.csv"
video_input_path = "C:\\Users\\WhuLi\\Documents\\Tmp\\圆桌派第四季20190829~3.mp4"
video_output_path = "C:\\Users\\WhuLi\\Documents\\Tmp\\"
t = 2


def fix_missing(inputs, duration):
    last = ' '
    output = []
    start = 0
    for i in inputs:
        if i["start"] > start:
            tmp = {
                "start": start,
                "end": i["start"],
                "name": search_f(inputs, inputs.index(i))
            }
            output.append(tmp)
            output.append(i)
            start = i['end']
            last = i['name']
        else:
            output.append(i)
            last = i['name']
    return output


def search_f(inputs, i):
    lenth = len(inputs) - i - 1
    if lenth <= 2:
        return inputs[i]["name"]
    for p in range(lenth):
        if inputs[i+p+1]["end"] - inputs[i+p+1]["start"] > 0.3:
            return inputs[i+p+1]["name"]
        elif inputs[i-p-1]["end"] - inputs[i-p-1]["start"] > 0.3:
            return inputs[i-p-1]["name"]
        if i-p-1 <= 0:
            return inputs[i]["name"]


def fix_wrong(inputs):
    outputs = []
    for i in range(len(inputs)):
        if i == 0:
            outputs.append(inputs[i])
            pass
        else:
            if inputs[i]["end"] - inputs[i]["start"] < 0.2:
                tmp = {
                    "start": inputs[i]["start"],
                    "end": inputs[i]["end"],
                    "name": search_f(inputs, i)
                }
                outputs.append(tmp)
            else:
                outputs.append(inputs[i])
    return outputs


def merge_timeline(time_line, names):
    output = []
    tmp = {}
    flag = 0
    for i in range(len(names)-1):
        if not flag:
            tmp = {
                "start": time_line[i],
                "end": time_line[i],
                "name": "speaker: "+names[i]
            }
            flag = 1
            continue
        else:
            if tmp["name"] != ("speaker: "+names[i]):
                output.append(tmp)
                tmp = {
                    "start": time_line[i],
                    "end": time_line[i],
                    "name": "speaker: "+names[i]
                }
                continue
            elif tmp["end"]+1 >= time_line[i]:
                tmp["end"] = time_line[i]
                continue
            else:
                output.append(tmp)
                tmp = {
                    "start": time_line[i],
                    "end": time_line[i],
                    "name": "speaker: "+names[i]
                }
                continue
    output.append(tmp)
    return output


def annotate(clip, txt, txt_color='red', fontsize=30, font='Xolonium-Bold'):
    txtclip = editor.TextClip(txt, fontsize=fontsize,
                              font=font, color=txt_color)
    cvc = editor.CompositeVideoClip(
        [clip, txtclip.set_pos(('center', 'top'))])
    return cvc.set_duration(clip.duration)


if __name__ == "__main__":
    df = np.loadtxt(subtitle_file_path, delimiter=',', dtype="str")
    time_line = np.array(df[:, 0], dtype="float")
    names = np.array(df[:, 1])

    video = editor.VideoFileClip(video_input_path)
    duration = video.duration

    output = merge_timeline(time_line, names)
    output = fix_wrong(output)
    output = fix_missing(output, duration)

    subs = []
    flag = 300
    count = 0
    mt = 0
    annotated_clips = []
    for s in tqdm(output):
        annotated_clips.append(
            annotate(video.subclip(s["start"], s["end"]), s["name"]))
        count += 1
        if count == flag:
            final_clip = editor.concatenate_videoclips(annotated_clips)
            final_clip.write_videofile(video_output_path+str(mt)+'_output.mp4')
            del annotated_clips
            gc.collect()
            annotated_clips = []
            mt += 1
            count = 0
    print("fin")
