import cv2
from deepface import DeepFace
import numpy as np  # just for proper typing


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


def run_emotion_model(img: np.ndarray) -> str:
    """Run the emotion detection backend with DeepFace for all faces in the image. Other backends can be added later on.

    Args:
        img (np.ndarray): Input image

    Returns:
        str: Emotion of the biggest face in the picture or None if no face was found
    """
    try:
        objs = DeepFace.analyze(
            img_path=img, actions=["age", "gender", "race", "emotion"]
        )
        areas = [get_face_area(obj["region"]) for obj in objs]
        biggest_face_idx = areas.index(max(areas))
        biggest_face_emotion = objs[biggest_face_idx]["dominant_emotion"]
        return biggest_face_emotion
    except ValueError:
        print(
            "Face could not be detected. Please confirm that the picture is a face photo."
        )
        return None


def get_emotion(url: str) -> str:
    """Analyze facial emotions with DeepFace for all faces in an image from a URL, only biggest face area is considered and returns its emotion.

    Args:
        url (str): Image URL to be processed.

    Raises:
        IOError: In case URL is not valid or doesn't contain an image

    Returns:
        str: Detected emotion of the biggest face in the image
    """
    success, img = get_image(url)
    if success:
        emotion = run_emotion_model(img)
        return emotion
    else:
        raise IOError("URL not valid or not an image", "URL is: {url}")
