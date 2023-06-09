import cv2
from deepface import DeepFace
import numpy as np  # just for proper typing


def get_image(url: str) -> tuple[cv2.VideoCapture, tuple[bool, np.ndarray]]:
    """Get a streamed image from a URL. WARNING! This will NOT work with video streams

    Args:
        url (str): image link to be processed

    Returns:
        tuple[cv2.VideoCapture, tuple[bool, np.ndarray]]: returns a VideoCapture object, check variable, and an image
    """
    cap = cv2.VideoCapture(url)
    return cap, cap.read()


def get_face_area(region: dict[str, int]) -> int:
    """claculate face area give its x,y,h,w region from DeepFace

    Args:
        region (dict[str, int]): a dictionary with x,y,w,h as keys

    Returns:
        int: returns face area
    """
    return region["w"] * region["h"]


def emotion_model(img: np.ndarray) -> tuple[bool, str]:
    """analyze facial emotions with DeepFace for all faces in the image, only biggest face area is considered.

    Args:
        img (np.ndarray): input image

    Returns:
        tuple[bool, str]: emotion of the biggest face in the picture
    """
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
    """Analyze facial emotions with DeepFace for all faces in an image from a URL, only biggest face area is considered and returns its emotion.

    Args:
        url (str): Image URL to be processed.

    Raises:
        IOError: In case URL is not valid or doesn't contain an image

    Returns:
        str: Detected emotion of the biggest face in the image
    """
    cap, (success, img) = get_image(url)
    cap.release()
    if success:
        found, emotion = emotion_model(img)
        if not found:
            return None
        return emotion
    else:
        raise IOError("URL not valid or not an image", "URL is: {url}")
