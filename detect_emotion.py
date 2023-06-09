import cv2
from deepface import DeepFace
import numpy as np  # just for proper typing


def get_image(url: str) -> tuple[cv2.VideoCapture, tuple[bool, np.ndarray]]:
    cap = cv2.VideoCapture(url)
    return cap, cap.read()


def get_face_area(region: dict[str, int]) -> int:
    return region["w"] * region["h"]


def emotion_model(img: np.ndarray) -> tuple[bool, str]:
    try:
        objs = DeepFace.analyze(
            img_path=img, actions=["age", "gender", "race", "emotion"]
        )
        areas = [get_face_area(obj["region"]) for obj in objs]
        biggest_face_idx = areas.index(max(areas))
        biggest_face_emotion = objs[biggest_face_idx]["dominant_emotion"]
        return True, biggest_face_emotion
    except ValueError:
        print(
            "Face could not be detected. Please confirm that the picture is a face photo."
        )
        return False, None


def get_emotion(url: str) -> str:
    cap, (success, img) = get_image(url)
    cap.release()
    if success:
        found, emotion = emotion_model(img)
        if not found:
            return None
        return emotion
    else:
        raise IOError("URL not valid or not an image", "URL is: {url}")
