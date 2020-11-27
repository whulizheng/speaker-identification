from cv2 import cv2 as cv2
from PIL import Image
import imagehash
import sys
import Utils
from moviepy.editor import VideoFileClip


def is_similar(img1, img2, cutoff=10, method="a"):
    img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    if method == 'a':
        n0 = imagehash.average_hash(img1)
        n1 = imagehash.average_hash(img2)
        p = n0 - n1
        if p < cutoff:
            return n0, n1, True
        else:
            return n0, n1, False
    else:
        print("Wrong method")
        exit(-1)


def get_points(clip, fps, cutoff):
    cut_points = [0]
    pre_img = None
    lenth = clip.duration*fps
    for i, img in enumerate(clip.iter_frames(fps)):
        if not i:
            pre_img = img
            continue
        _, _, result = is_similar(pre_img, img, cutoff)
        pre_img = img
        if not result:
            cut_points.append((i) / fps)
        sys.stdout.write('\r检索中：'+str(round((i/lenth)*100, 2))+"%|100%")
        sys.stdout.flush()
    cut_points.append(clip.duration)
    return cut_points


def cut_save_video(clip, start_time, end_time, name):
    if start_time == end_time:
        return False
    else:
        clip.subclip(start_time, end_time).write_videofile(name)
        return True


if __name__ == "__main__":
    clip = VideoFileClip(
        "C:\\Users\\WhuLi\\Documents\\Tmp\\圆桌派第四季20190829.mp4")
    config = Utils.readjson("config.json")
    cut_points = get_points(clip, clip.fps, config["video"]["cut_threshold"])
    for i in range(len(cut_points)-1):
        cut_save_video(
            clip, cut_points[i], cut_points[i+1], "results\\"+str(i)+".mp4")
