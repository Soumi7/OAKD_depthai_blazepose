import cv2
from math import atan2, degrees
import sys
sys.path.append("../..")
from BlazeposeDepthaiEdge import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
from mediapipe_utils import KEYPOINT_DICT
import argparse
import numpy as np

import argparse
import csv
import os
from math import acos, atan2
from pathlib import Path

class EMADictSmoothing(object):
    """Smoothes pose classification."""

    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        """Smoothes given pose classification.

    Smoothing is done by computing Exponential Moving Average for every pose
    class observed in the given time window. Missed pose classes arre replaced
    with 0.

    Args:
      data: Dictionary with pose classification. Sample:
          {
            'pushups_down': 8,
            'pushups_up': 2,
          }

    Result:
      Dictionary in the same format but with smoothed and float instead of
      integer values. Sample:
        {
          'pushups_down': 8.3,
          'pushups_up': 1.7,
        }
    """
        # Add new data to the beginning of the window for simpler code.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[:self._window_size]

        # Get all keys.
        keys = set(
            [key for data in self._data_in_window for key, _ in data.items()])

        # Get smoothed values.
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # Update factor.
                factor *= (1.0 - self._alpha)

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data


class PoseSample(object):

    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name

        self.embedding = embedding


class PoseSampleOutlier(object):

    def __init__(self, sample, detected_class, all_classes):
        self.sample = sample
        self.detected_class = detected_class
        self.all_classes = all_classes


class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding

    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances defined in `_get_pose_distance_embedding`.
    """
        assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(
            landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it easier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index(
            'right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

    We use several pairwise 3D distances to form pose embedding. All distances
    include X and Y components with sign. We differnt types of pairs to cover
    different pose classes. Feel free to remove some or add new.

    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances.
    """
        embedding = np.array([
            # One joint.

            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_elbow'),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_elbow'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            self._get_distance_by_names(
                landmarks, 'right_elbow', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(
                landmarks, 'right_knee', 'right_ankle'),

            # Two joints.

            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_wrist'),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.

            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.

            self._get_distance_by_names(
                landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            self._get_distance_by_names(
                landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(
                landmarks, 'left_ankle', 'right_ankle'),

            # Body bent direction.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from


class PoseClassifier(object):
    """Classifies pose landmarks."""

    def __init__(self,
                 pose_samples_folder,
                 pose_embedder,
                 file_extension='csv',
                 file_separator=',',
                 n_landmarks=33,
                 n_dimensions=3,
                 top_n_by_max_distance=30,
                 top_n_by_mean_distance=10,
                 axes_weights=(1., 1., 0.2)):
        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
        self._axes_weights = axes_weights

        self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                     file_extension,
                                                     file_separator,
                                                     n_landmarks,
                                                     n_dimensions,
                                                     pose_embedder)

    def _load_pose_samples(self,
                           pose_samples_folder,
                           file_extension,
                           file_separator,
                           n_landmarks,
                           n_dimensions,
                           pose_embedder):
        """Loads pose samples from a given folder.

    Required folder structure:
      neutral_standing.csv
      pushups_down.csv
      pushups_up.csv
      squats_down.csv
      ...

    Required CSV structure:
      sample_00001,x1,y1,z1,x2,y2,z2,....
      sample_00002,x1,y1,z1,x2,y2,z2,....
      ...
    """
        # Each file in the folder represents one pose class.
        file_names = [name for name in os.listdir(
            pose_samples_folder) if name.endswith(file_extension)]

        pose_samples = []
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[:-(len(file_extension) + 1)]

            # Parse CSV.
            with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=file_separator)
                for row in csv_reader:
                    assert len(row) == n_landmarks * n_dimensions + \
                        1, 'Wrong number of values: {}'.format(len(row))
                    landmarks = np.array(row[1:], np.float32).reshape(
                        [n_landmarks, n_dimensions])
                    pose_samples.append(PoseSample(
                        name=row[0],
                        landmarks=landmarks,
                        class_name=class_name,
                        embedding=pose_embedder(landmarks),
                    ))

        return pose_samples

    def find_pose_sample_outliers(self):
        """Classifies each sample against the entire database."""
        # Find outliers in target poses
        outliers = []
        for sample in self._pose_samples:
            # Find nearest poses for the target one.
            pose_landmarks = sample.landmarks.copy()
            pose_classification = self.__call__(pose_landmarks)
            class_names = [class_name for class_name, count in pose_classification.items() if
                           count == max(pose_classification.values())]

            # Sample is an outlier if nearest poses have different class or more than
            # one pose class is detected as nearest.
            if sample.class_name not in class_names or len(class_names) != 1:
                outliers.append(PoseSampleOutlier(
                    sample, class_names, pose_classification))

        return outliers

    def __call__(self, pose_landmarks):
        """Classifies given pose.

    Classification is done in two stages:
      * First we pick top-N samples by MAX distance. It allows to remove samples
        that are almost the same as given pose, but has few joints bent in the
        other direction.
      * Then we pick top-N samples by MEAN distance. After outliers are removed
        on a previous step, we can pick samples that are closes on average.

    Args:
      pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

    Returns:
      Dictionary with count of nearest pose samples from the database. Sample:
        {
          'pushups_down': 8,
          'pushups_up': 2,
        }
    """
        # Check that provided and target poses have the same shape.
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(
            pose_landmarks.shape)

        # Get given pose embedding.
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(
            pose_landmarks * np.array([-1, 1, 1]))

        # Filter by max distance.
        #
        # That helps to remove outliers - poses that are almost the same as the
        # given one, but has one joint bent into another direction and actually
        # represnt a different pose class.
        max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding)
                       * self._axes_weights),
                np.max(np.abs(sample.embedding - flipped_pose_embedding)
                       * self._axes_weights),
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        # Filter by mean distance.
        #
        # After removing outliers we can find the nearest pose by mean distance.

        # HERE2
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding)
                        * self._axes_weights),
                np.mean(np.abs(sample.embedding - flipped_pose_embedding)
                        * self._axes_weights),
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

        # print(mean_dist_heap[0])
        # print(mean_dist_heap)
        # for _, k in mean_dist_heap:
        #     print(self._pose_samples[k].class_name)

        # Collect results into map: (class_name -> n_samples)
        class_names = [
            self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        result = {class_name: class_names.count(
            class_name) for class_name in set(class_names)}

        # print(result)

        return result


# LINES_*_BODY are used when drawing the skeleton onto the source image.
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
LINES_FULL_BODY = [[28, 30, 32, 28, 26, 24, 12, 11, 23, 25, 27, 29, 31, 27],
                   [23, 24],
                   [22, 16, 18, 20, 16, 14, 12],
                   [21, 15, 17, 19, 15, 13, 11],
                   [8, 6, 5, 4, 0, 1, 2, 3, 7],
                   [10, 9],
                   ]
LINES_UPPER_BODY = [[12, 11, 23, 24, 12],
                    [22, 16, 18, 20, 16, 14, 12],
                    [21, 15, 17, 19, 15, 13, 11],
                    [8, 6, 5, 4, 0, 1, 2, 3, 7],
                    [10, 9],
                    ]
# LINE_MESH_*_BODY are used when drawing the skeleton in 3D.
rgb = {"right": (0, 1, 0), "left": (1, 0, 0), "middle": (1, 1, 0)}
LINE_MESH_FULL_BODY = [[9, 10], [4, 6], [1, 3],
                       [12, 14], [14, 16], [16, 20], [20, 18], [18, 16],
                       [12, 11], [11, 23], [23, 24], [24, 12],
                       [11, 13], [13, 15], [15, 19], [19, 17], [17, 15],
                       [24, 26], [26, 28], [32, 30],
                       [23, 25], [25, 27], [29, 31]]
LINE_TEST = [[12, 11], [11, 23], [23, 24], [24, 12]]

COLORS_FULL_BODY = ["middle", "right", "left",
                    "right", "right", "right", "right", "right",
                    "middle", "middle", "middle", "middle",
                    "left", "left", "left", "left", "left",
                    "right", "right", "right", "left", "left", "left"]
COLORS_FULL_BODY = [rgb[x] for x in COLORS_FULL_BODY]
LINE_MESH_UPPER_BODY = [[9, 10], [4, 6], [1, 3],
                        [12, 14], [14, 16], [16, 20], [20, 18], [18, 16],
                        [12, 11], [11, 23], [23, 24], [24, 12],
                        [11, 13], [13, 15], [15, 19], [19, 17], [17, 15]
                        ]

def recognize_pose(r):
        assert r.landmarks.shape == (
            33, 3), 'Unexpected landmarks shape: {}'.format(r.landmarks.shape)

        r.landmarks = r.landmarks.astype('float32')
        keypoint_json = "{\'landmarks\':["

        for key in KEYPOINT_DICT.keys():
                coord = r.landmarks[KEYPOINT_DICT[key]]
                keypoint_json+= f'{{\'name\':\'{key}\',\'x\':{coord[0]},\'y\':{coord[1]}}},'  

        keypoint_json = keypoint_json[:-1] + ']}'
        print(f'LANDMARKS: {keypoint_json}')

        #data = {"pose": pose, "accuracy": rounded_accuracy, "feedback": feedback}
        #print(f"RECOGNIZED: {data}")

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['full', 'lite', '831'], default='lite',
                        help="Landmark model to use (default=%(default)s")
parser.add_argument('-i', '--input', type=str, default='rgb_laconic',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")  
parser.add_argument("-o","--output",
                    help="Path to output video file")

args = parser.parse_args()            

pose = BlazeposeDepthai(input_src=args.input, lm_model=args.model)

while True:
    # Run blazepose on next frame
    frame, body = pose.next_frame()
    if frame is None: break
    if body: 
        predicted_pose = recognize_pose(body)
    for i in range(0,33):
        print(KEYPOINT_DICT[i],body.is_present(i))
        
pose.exit()

