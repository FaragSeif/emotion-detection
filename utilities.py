import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace


DF = DeepFace.analyze
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = list(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE = list(mp_face_mesh.FACEMESH_RIGHT_EYE)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
left_eyelids = [
    LEFT_EYE[0][0],
    LEFT_EYE[5][0],
    LEFT_EYE[10][0],
    LEFT_EYE[-1][1],
]
right_eyelids = [
    RIGHT_EYE[14][0],
    RIGHT_EYE[12][0],
    RIGHT_EYE[3][0],
    RIGHT_EYE[-1][1],
]


def get_image(url: str) -> tuple[bool, np.ndarray]:
    """Get a streamed image from a URL. WARNING! This will NOT work with video streams

    Args:
        url (str): Image link to be processed

    Returns:
        tuple[bool, np.ndarray]: Returns a check variable and an image
    """
    cap = cv2.VideoCapture(url)
    success, img = cap.read()
    cap.release()
    return success, img


def get_face_area(region: dict[str, int]) -> int:
    """Calculate face area give its x,y,h,w region from DeepFace

    Args:
        region (dict[str, int]): A dictionary with x,y,w,h as keys

    Returns:
        int: Returns face area
    """
    return region["w"] * region["h"]


def get_hue(img, coordinates):
    x, y = coordinates
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv_img[x, y, 0]


def calculate_distance(normalized_point_list):
    left_v_dist = (
        normalized_point_list[left_eyelids[0]][1]
        - normalized_point_list[left_eyelids[2]][1]
    )
    left_h_dist = (
        normalized_point_list[left_eyelids[1]][0]
        - normalized_point_list[left_eyelids[3]][0]
    )

    right_v_dist = (
        normalized_point_list[right_eyelids[0]][1]
        - normalized_point_list[right_eyelids[2]][1]
    )
    right_h_dist = (
        normalized_point_list[right_eyelids[3]][0]
        - normalized_point_list[right_eyelids[1]][0]
    )

    left_distance = (left_v_dist, left_h_dist)
    right_distance = (right_v_dist, right_h_dist)

    return [left_distance, right_distance]


def calculate_iris_radius(normalized_point_list):
    (left_iris_cx, left_iris_cy), left_iris_r = cv2.minEnclosingCircle(
        normalized_point_list[LEFT_IRIS]
    )
    (right_iris_cx, right_iris_cy), right_iris_r = cv2.minEnclosingCircle(
        normalized_point_list[RIGHT_IRIS]
    )

    left_iris_center = [int(left_iris_cx), int(left_iris_cy)]
    right_iris_center = [int(right_iris_cx), int(right_iris_cy)]

    return [
        (left_iris_center, int(left_iris_r)),
        (right_iris_center, int(right_iris_r)),
    ]
