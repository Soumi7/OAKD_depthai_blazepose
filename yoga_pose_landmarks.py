import cv2
from math import atan2, degrees
import sys
sys.path.append("../..")
from mediapipe_utils import KEYPOINT_DICT
import argparse
import numpy as np

import argparse
import csv
import os
from math import acos, atan2
from pathlib import Path

def recognize_pose(r):

        print(f"RECOGNIZED: {r.landmarks}")

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['full', 'lite', '831'], default='lite',
                        help="Landmark model to use (default=%(default)s")
parser.add_argument('-i', '--input', type=str, default='rgb_laconic',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")  
parser.add_argument("-o","--output",
                    help="Path to output video file")
args = parser.parse_args()            

while True:
    # Run blazepose on next frame
    frame, body = pose.next_frame()
    if frame is None: break
    if body: 
        predicted_pose = recognize_pose(body)
pose.exit()

