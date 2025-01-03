from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

from tqdm import tqdm
import cv2
import os
from os.path import join


def extract_frames_from_video(src_path, dist_path):
    video_capture = cv2.VideoCapture(src_path)
    # Trying to read the first frames
    success, image = video_capture.read()

    frame_number = 0
    os.makedirs(dist_path)
    while success:
        cv2.imwrite(join(dist_path, f"frame{frame_number}.jpg"), image)
        success, image = video_capture.read()
        frame_number += 1


dataset_path = "E:\\IPRIA Datasets\\konvid1k\\videos"
dist_path = "E:\\IPRIA Datasets\\konvid1k\\frames"

for video in os.listdir(dataset_path):
    video_path = os.path.join(dataset_path, video)
    extract_frames_from_video(
        video_path, os.path.join(dist_path, video.replace(".mp4", ""))
    )
