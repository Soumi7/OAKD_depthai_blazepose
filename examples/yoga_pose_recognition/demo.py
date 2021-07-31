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


POSES = {
    "mountain": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 90,
        "RIGHT_HAND_HIP_ANGLE": 90,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 180,
        "RIGHT_HIP_KNEE_ANGLE": 180,

        "ANGLE_BETWEEN_LEGS": 0,
    },
    "tree": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 180,
        "RIGHT_HAND_HIP_ANGLE": 180,

        "LEFT_LEG_ANGLE": 135,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 0,
        "RIGHT_HIP_KNEE_ANGLE": 180,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "boat": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 45,
        "RIGHT_HAND_HIP_ANGLE": 45,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 180,
        "RIGHT_HIP_KNEE_ANGLE": 180,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "bridge": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 45,
        "RIGHT_HAND_HIP_ANGLE": 45,

        "LEFT_LEG_ANGLE": 90,
        "RIGHT_LEG_ANGLE": 90,

        "LEFT_HIP_KNEE_ANGLE": 180,
        "RIGHT_HIP_KNEE_ANGLE": 180,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "butterfly": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 0,
        "RIGHT_HAND_HIP_ANGLE": 0,

        "LEFT_LEG_ANGLE": 20,
        "RIGHT_LEG_ANGLE": 20,

        "LEFT_HIP_KNEE_ANGLE": 90,
        "RIGHT_HIP_KNEE_ANGLE": 90,

        "ANGLE_BETWEEN_LEGS": 180,
    },

    "camel": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 90,
        "RIGHT_HAND_HIP_ANGLE": 90,

        "LEFT_LEG_ANGLE": 90,
        "RIGHT_LEG_ANGLE": 90,

        "LEFT_HIP_KNEE_ANGLE": 180,
        "RIGHT_HIP_KNEE_ANGLE": 180,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "cat_cow": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 90,
        "RIGHT_HAND_HIP_ANGLE": 90,

        "LEFT_LEG_ANGLE": 90,
        "RIGHT_LEG_ANGLE": 90,

        "LEFT_HIP_KNEE_ANGLE": 90,
        "RIGHT_HIP_KNEE_ANGLE": 90,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "chair": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 180,
        "RIGHT_HAND_HIP_ANGLE": 180,

        "LEFT_LEG_ANGLE": 100,
        "RIGHT_LEG_ANGLE": 100,

        "LEFT_HIP_KNEE_ANGLE": 80,
        "RIGHT_HIP_KNEE_ANGLE": 80,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "child": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 170,
        "RIGHT_HAND_HIP_ANGLE": 170,

        "LEFT_LEG_ANGLE": 10,
        "RIGHT_LEG_ANGLE": 10,

        "LEFT_HIP_KNEE_ANGLE": 10,
        "RIGHT_HIP_KNEE_ANGLE": 10,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "cobra": {
        "LEFT_ARM_ANGLE": 120,
        "RIGHT_ARM_ANGLE": 120,

        "LEFT_HAND_HIP_ANGLE": 40,
        "RIGHT_HAND_HIP_ANGLE": 40,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 180,
        "RIGHT_HIP_KNEE_ANGLE": 180,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "corpse": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 40,
        "RIGHT_HAND_HIP_ANGLE": 40,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 180,
        "RIGHT_HIP_KNEE_ANGLE": 180,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "cow_face": {
        "LEFT_ARM_ANGLE": 10,
        "RIGHT_ARM_ANGLE": 10,

        "LEFT_HAND_HIP_ANGLE": 180,
        "RIGHT_HAND_HIP_ANGLE": 180,

        "LEFT_LEG_ANGLE": 90,
        "RIGHT_LEG_ANGLE": 90,

        "LEFT_HIP_KNEE_ANGLE": 80,
        "RIGHT_HIP_KNEE_ANGLE": 80,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "downward_facing_dog": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 180,
        "RIGHT_HAND_HIP_ANGLE": 180,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 80,
        "RIGHT_HIP_KNEE_ANGLE": 80,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "easy": {
        "LEFT_ARM_ANGLE": 100,
        "RIGHT_ARM_ANGLE": 100,

        "LEFT_HAND_HIP_ANGLE": 20,
        "RIGHT_HAND_HIP_ANGLE": 20,

        "LEFT_LEG_ANGLE": 10,
        "RIGHT_LEG_ANGLE": 10,

        "LEFT_HIP_KNEE_ANGLE": 90,
        "RIGHT_HIP_KNEE_ANGLE": 90,

        "ANGLE_BETWEEN_LEGS": 120,
    },

    "fish": {
        "LEFT_ARM_ANGLE": 120,
        "RIGHT_ARM_ANGLE": 120,

        "LEFT_HAND_HIP_ANGLE": 40,
        "RIGHT_HAND_HIP_ANGLE": 40,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 120,
        "RIGHT_HIP_KNEE_ANGLE": 120,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "forward_bend": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 120,
        "RIGHT_HAND_HIP_ANGLE": 120,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 0,
        "RIGHT_HIP_KNEE_ANGLE": 0,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "half_moon": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 90,
        "RIGHT_HAND_HIP_ANGLE": 90,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 90,
        "RIGHT_HIP_KNEE_ANGLE": 180,

        "ANGLE_BETWEEN_LEGS": 90,
    },

    "half_spinal": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 80,

        "LEFT_HAND_HIP_ANGLE": 45,
        "RIGHT_HAND_HIP_ANGLE": 45,

        "LEFT_LEG_ANGLE": 45,
        "RIGHT_LEG_ANGLE": 45,

        "LEFT_HIP_KNEE_ANGLE": 45,
        "RIGHT_HIP_KNEE_ANGLE": 90,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "legs_up_the_wall": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 90,
        "RIGHT_HAND_HIP_ANGLE": 90,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 90,
        "RIGHT_HIP_KNEE_ANGLE": 90,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "locust": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 0,
        "RIGHT_HAND_HIP_ANGLE": 0,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 120,
        "RIGHT_HIP_KNEE_ANGLE": 120,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "plank": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 70,
        "RIGHT_HAND_HIP_ANGLE": 70,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 180,
        "RIGHT_HIP_KNEE_ANGLE": 180,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "ragdoll": {
        "LEFT_ARM_ANGLE": 90,
        "RIGHT_ARM_ANGLE": 90,

        "LEFT_HAND_HIP_ANGLE": 140,
        "RIGHT_HAND_HIP_ANGLE": 140,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 10,
        "RIGHT_HIP_KNEE_ANGLE": 10,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "seated_forward_bend": {
        "LEFT_ARM_ANGLE": 90,
        "RIGHT_ARM_ANGLE": 90,

        "LEFT_HAND_HIP_ANGLE": 120,
        "RIGHT_HAND_HIP_ANGLE": 120,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 0,
        "RIGHT_HIP_KNEE_ANGLE": 0,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "seated_wide_angle": {
        "LEFT_ARM_ANGLE": 90,
        "RIGHT_ARM_ANGLE": 90,

        "LEFT_HAND_HIP_ANGLE": 40,
        "RIGHT_HAND_HIP_ANGLE": 40,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 90,
        "RIGHT_HIP_KNEE_ANGLE": 90,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "staff": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 90,
        "RIGHT_HAND_HIP_ANGLE": 90,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 90,
        "RIGHT_HIP_KNEE_ANGLE": 90,

        "ANGLE_BETWEEN_LEGS": 0,
    },

    "tree": {
        "LEFT_ARM_ANGLE": 45,
        "RIGHT_ARM_ANGLE": 45,

        "LEFT_HAND_HIP_ANGLE": 45,
        "RIGHT_HAND_HIP_ANGLE": 45,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 40,

        "LEFT_HIP_KNEE_ANGLE": 180,
        "RIGHT_HIP_KNEE_ANGLE": 120,

        "ANGLE_BETWEEN_LEGS": 45,
    },

    "triangle1": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 90,
        "RIGHT_HAND_HIP_ANGLE": 90,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 120,
        "RIGHT_HIP_KNEE_ANGLE": 45,

        "ANGLE_BETWEEN_LEGS": 90,
    },

    "triangle2": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 90,
        "RIGHT_HAND_HIP_ANGLE": 90,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 45,
        "RIGHT_HIP_KNEE_ANGLE": 120,

        "ANGLE_BETWEEN_LEGS": 90,
    },

    "warrior": {
        "LEFT_ARM_ANGLE": 180,
        "RIGHT_ARM_ANGLE": 180,

        "LEFT_HAND_HIP_ANGLE": 90,
        "RIGHT_HAND_HIP_ANGLE": 90,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 90,

        "LEFT_HIP_KNEE_ANGLE": 120,
        "RIGHT_HIP_KNEE_ANGLE": 90,

        "ANGLE_BETWEEN_LEGS": 120,
    },

    "wide-legged_forward_bend": {
        "LEFT_ARM_ANGLE": 90,
        "RIGHT_ARM_ANGLE": 90,

        "LEFT_HAND_HIP_ANGLE": 90,
        "RIGHT_HAND_HIP_ANGLE": 90,

        "LEFT_LEG_ANGLE": 180,
        "RIGHT_LEG_ANGLE": 180,

        "LEFT_HIP_KNEE_ANGLE": 45,
        "RIGHT_HIP_KNEE_ANGLE": 45,

        "ANGLE_BETWEEN_LEGS": 90,
    },
}


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
        # Multiplication by 100 is not required, but makes it eaasier to debug.
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


# For gesture demo
semaphore_flag = {
        (3,4):'A', (2,4):'B', (1,4):'C', (0,4):'D',
        (4,7):'E', (4,6):'F', (4,5):'G', (2,3):'H',
        (0,3):'I', (0,6):'J', (3,0):'K', (3,7):'L',
        (3,6):'M', (3,5):'N', (2,1):'O', (2,0):'P',
        (2,7):'Q', (2,6):'R', (2,5):'S', (1,0):'T',
        (1,7):'U', (0,5):'V', (7,6):'W', (7,5):'X',
        (1,6):'Y', (5,6):'Z',
}

def recognize_pose(r, expected_pose="mountain", track="beginners"):

        r.pose = "Pose not detected"

        #################################################################################

        pose_embedder = FullBodyPoseEmbedder()

        if track == "beginners":
            pose_folder = "./pose_csvs/beginners_poses_csvs_out"
        elif track == "asthma":
            pose_folder = "./pose_csvs/asthma_poses_csvs_out"
        elif track == "power":
            pose_folder = "./pose_csvs/power_poses_csvs_out"
        elif track == "immunity":
            pose_folder = "./pose_csvs/immunity_poses_csvs_out"
        elif track == "insomnia":
            pose_folder = "./pose_csvs/insomnia_poses_csvs_out"
        elif track == "cardiovascular":
            pose_folder = "./pose_csvs/cardiovascular_poses_csvs_out"
        elif track == "migraine":
            pose_folder = "./pose_csvs/migraine_poses_csvs_out"
        elif track == "pregnancy":
            pose_folder = "./pose_csvs/pregnancy_poses_csvs_out"

        pose_classifier = PoseClassifier(
            pose_samples_folder=pose_folder,
            pose_embedder=pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)

        print(r.landmarks)
        print(r.landmarks[0])

        assert r.landmarks.shape == (
            33, 3), 'Unexpected landmarks shape: {}'.format(r.landmarks.shape)

        # print(r.landmarks_abs)
        # print(type(r.landmarks_abs))

        r.landmarks = r.landmarks.astype('float32')

        pose_classification = pose_classifier(r.landmarks)

        pose_classification_filter = EMADictSmoothing(
            window_size=10,
            alpha=0.2)

        # Smooth classification using EMA.
        pose_classification_filtered = pose_classification_filter(
            pose_classification)

        max_sample = 0
        pose = 0

        # print(pose_classification_filtered)

        for i in pose_classification_filtered.keys():
            if pose_classification_filtered[i] > max_sample:
                pose = i
                max_sample = pose_classification_filtered[i]

        r.pose = pose

        accuracy = max_sample/10

        # data = {"pose": pose, "accuracy": rounded_accuracy}

        # value = db.child("123").get()
        # if value.val() is None:
        #     db.child("123").set(data)
        # else:
        #     db.child("123").update(data)

        # def getAngle(firstPoint, midPoint, lastPoint):
        #     result = np.degrees(atan2(lastPoint[1] - midPoint[1],lastPoint[0] - midPoint[0])
        #         - atan2(firstPoint[1] - midPoint[1], firstPoint[0] - midPoint[0]))
        #     result = abs(result) # Angle should never be negative
        #     if (result > 180) :
        #         result = 360.0 - result # Always get the acute representation of the angle

        #         result = 360.0 - result # Always get the acute representation of the angle
        #     return result
        # print(r.landmarks_abs[14,:2])
        # print(r.landmarks_abs[14])
        # print(r.landmarks_abs[14,:3])

        def get3DAngle(A, B, C):
            # v1 = {A[0] - B[0], A[1] - B[1], A[2] - B[2]}
            # v2 = {C[0] - B[0], C[1] - B[1], C[2] - B[2]}
            # v1mag = (A[0] * A[0] + A[1] * A[1] + A[2] * A[2])**(1/2)
            # v1norm = {A[0] / v1mag, A[1] / v1mag, A[2] / v1mag}
            # v2mag = (B[0] * B[0] + B[1] * B[1] + B[2] * B[2])**(1/2)
            # v2norm = {B[0] / v2mag, B[1] / v2mag, B[2] / v2mag}
            # res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
            # angle = acos(res)
            a = np.array(A)
            b = np.array(B)
            c = np.array(C)

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / \
                (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)

        LEFT_ARM_ANGLE = get3DAngle(
            r.landmarks[12, :3], r.landmarks[14, :3], r.landmarks[16, :3])
        RIGHT_ARM_ANGLE = get3DAngle(
            r.landmarks[11, :3], r.landmarks[13, :3], r.landmarks[15, :3])

        LEFT_HAND_HIP_ANGLE = get3DAngle(
            r.landmarks[14, :3], r.landmarks[12, :3], r.landmarks[24, :3])
        RIGHT_HAND_HIP_ANGLE = get3DAngle(
            r.landmarks[13, :3], r.landmarks[21, :3], r.landmarks[23, :3])

        LEFT_LEG_ANGLE = get3DAngle(
            r.landmarks[24, :3], r.landmarks[26, :3], r.landmarks[28, :3])
        RIGHT_LEG_ANGLE = get3DAngle(
            r.landmarks[23, :3], r.landmarks[25, :3], r.landmarks[27, :3])

        LEFT_HIP_KNEE_ANGLE = get3DAngle(
            r.landmarks[12, :3], r.landmarks[24, :3], r.landmarks[26, :3])
        RIGHT_HIP_KNEE_ANGLE = get3DAngle(
            r.landmarks[11, :3], r.landmarks[23, :3], r.landmarks[25, :3])

        ANGLE_BETWEEN_LEGS = get3DAngle(
            r.landmarks[26, :3], r.landmarks[0, :3], r.landmarks[25, :3])

        # print("LEFT_ARM_ANGLE",LEFT_ARM_ANGLE)
        # print("RIGHT_ARM_ANGLE",RIGHT_ARM_ANGLE)

        # print("LEFT_HAND_HIP_ANGLE",LEFT_HAND_HIP_ANGLE)
        # print("RIGHT_HAND_HIP_ANGLE",RIGHT_HAND_HIP_ANGLE)

        # print("LEFT_LEG_ANGLE",LEFT_LEG_ANGLE)
        # print("RIGHT_LEG_ANGLE",RIGHT_LEG_ANGLE)

        # print("LEFT_HIP_KNEE_ANGLE", LEFT_HIP_KNEE_ANGLE)
        # print("RIGHT_HIP_KNEE_ANGLE", RIGHT_HIP_KNEE_ANGLE)

        # print("ANGLE_BETWEEN_LEGS", ANGLE_BETWEEN_LEGS)
        from collections import OrderedDict
        diff_dict = OrderedDict()
        feedback = ""
        if expected_pose == "mountain":
            pose_angles = POSES["mountain"]
        elif expected_pose == "tree":
            pose_angles = POSES["tree"]
        elif expected_pose == "downwarddog":
            pose_angles = POSES["downwarddog"]
        elif expected_pose == "child":
            pose_angles = POSES["child"]
        elif expected_pose == "boat":
            pose_angles = POSES["boat"]
        elif expected_pose == "bridge":
            pose_angles = POSES["bridge"]
        elif expected_pose == "butterfly":
            pose_angles = POSES["butterfly"]
        elif expected_pose == "camel":
            pose_angles = POSES["camel"]
        elif expected_pose == "cat_cow":
            pose_angles = POSES["cat_cow"]
        elif expected_pose == "chair":
            pose_angles = POSES["chair"]
        elif expected_pose == "cobra":
            pose_angles = POSES["cobra"]
        elif expected_pose == "corpse":
            pose_angles = POSES["corpse"]
        elif expected_pose == "cow_face":
            pose_angles = POSES["cow_face"]
        elif expected_pose == "downward_facing_dog":
            pose_angles = POSES["downward_facing_dog"]
        elif expected_pose == "easy":
            pose_angles = POSES["easy"]
        elif expected_pose == "fish":
            pose_angles = POSES["fish"]
        elif expected_pose == "child":
            pose_angles = POSES["child"]
        elif expected_pose == "forward_bend":
            pose_angles = POSES["forward_bend"]
        elif expected_pose == "half_moon":
            pose_angles = POSES["half_moon"]
        elif expected_pose == "half_spinal":
            pose_angles = POSES["half_spinal"]
        elif expected_pose == "head_to_knee":
            pose_angles = POSES["head_to_knee"]
        elif expected_pose == "hypnotic_sphinx":
            pose_angles = POSES["hypnotic_sphinx"]
        elif expected_pose == "leg_up_the_wall":
            pose_angles = POSES["leg_up_the_wall"]
        elif expected_pose == "locust":
            pose_angles = POSES["locust"]
        elif expected_pose == "plank":
            pose_angles = POSES["plank"]
        elif expected_pose == "ragdoll":
            pose_angles = POSES["ragdoll"]
        elif expected_pose == "seated_forward_bend":
            pose_angles = POSES["seated_forward_bend"]
        elif expected_pose == "seated_wide_angle":
            pose_angles = POSES["seated_wide_angle"]
        elif expected_pose == "staff":
            pose_angles = POSES["staff"]
        elif expected_pose == "triangle1":
            pose_angles = POSES["triangle1"]
        elif expected_pose == "triangle2":
            pose_angles = POSES["triangle2"]
        elif expected_pose == "warrior":
            pose_angles = POSES["warrior"]
        elif expected_pose == "wide-legged_forward_bend":
            pose_angles = POSES["wide-legged_forward_bend"]

        diff_dict["LEFT_ARM_ANGLE"] = pose_angles["LEFT_ARM_ANGLE"] - \
            LEFT_ARM_ANGLE
        diff_dict["RIGHT_ARM_ANGLE"] = pose_angles["RIGHT_ARM_ANGLE"] - \
            RIGHT_ARM_ANGLE
        diff_dict["LEFT_HAND_HIP_ANGLE"] = pose_angles["LEFT_HAND_HIP_ANGLE"] - \
            LEFT_HAND_HIP_ANGLE
        diff_dict["RIGHT_HAND_HIP_ANGLE"] = pose_angles["RIGHT_HAND_HIP_ANGLE"] - \
            RIGHT_HAND_HIP_ANGLE
        diff_dict["RIGHT_LEG_ANGLE"] = pose_angles["RIGHT_LEG_ANGLE"] - \
            RIGHT_LEG_ANGLE
        diff_dict["LEFT_HIP_KNEE_ANGLE"] = pose_angles["LEFT_HIP_KNEE_ANGLE"] - \
            LEFT_HIP_KNEE_ANGLE
        diff_dict["RIGHT_HIP_KNEE_ANGLE"] = pose_angles["LEFT_HIP_KNEE_ANGLE"] - \
            RIGHT_HIP_KNEE_ANGLE
        diff_dict["ANGLE_BETWEEN_LEGS"] = pose_angles["ANGLE_BETWEEN_LEGS"] - \
            ANGLE_BETWEEN_LEGS

        diff_dict = sorted(diff_dict.items(),
                           key=lambda item: abs(item[1]), reverse=True)
        # print(diff_dict)

        new_accuracy = 0
        accuracy_threshold = 180

        feedback = "{"

        # jointname1 _positive:jointname_name#
        for key in diff_dict[0:2]:
            # feedback += key[0]+":"+str(key[1])+"#"
            # feedback += "\'" + key[0] + "\':" + key[1] + ","
            value = key[1]
            feedback += f'\'{key[0]}\':{value:.2f},'
        
        feedback = feedback[:-1] + "}"
             

        if pose == expected_pose:
            for key in diff_dict:
                calculated_accuracy = 1 - (abs(key[1]) / accuracy_threshold)
                new_accuracy += calculated_accuracy

            new_accuracy /= len(diff_dict)

            # calculating weighted average
            # giving more weightage to classes
            # less weightage to angles
            weighted_accuracy = accuracy * 0.6 + new_accuracy * 0.4
            rounded_accuracy = round(weighted_accuracy, 2)

            # if pose == "triangle1" or pose == "triangle2":
            #     pose = "triangle"

            data = {"pose": pose, "accuracy": rounded_accuracy, "feedback": feedback}
            print(f"RECOGNIZED: {data}")

            # print("----------------------")
            # print(f'POSE: {pose}')
            # print(f'ACCURACY: classes: {accuracy}, angles: {new_accuracy}')
            # print(f'WEIGHTED: {weighted_accuracy}')
            # print(f'FEEDBACK: {feedback}')
            # print("----------------------\n")

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['full', 'lite', '831'], default='full',
                        help="Landmark model to use (default=%(default)s")
parser.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")  
parser.add_argument("-o","--output",
                    help="Path to output video file")
parser.add_argument('-ep', '--expected_pose', type=str,
                    help="enable pose recognition")
parser.add_argument('-tr', '--track', type=str,
                    help="select specific track")
args = parser.parse_args()            

pose = BlazeposeDepthai(input_src=args.input, lm_model=args.model)
renderer = BlazeposeRenderer(pose, output=args.output)
expected_pose = args.expected_pose
track = args.track

while True:
    # Run blazepose on next frame
    frame, body = pose.next_frame()
    if frame is None: break
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    # Gesture recognition
    if body: 
        predicted_pose = recognize_pose(body,expected_pose, track)
        if predicted_pose:
            cv2.putText(frame, predicted_pose, (frame.shape[1] // 2, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0,190,255), 3)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
pose.exit()

