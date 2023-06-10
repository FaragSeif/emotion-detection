import numpy as np
from utilities import *


def run_emotion_model(img: np.ndarray) -> str:
    """Run the emotion detection backend with DeepFace for all faces in the image. Other backends can be added later on.

    Args:
        img (np.ndarray): Input image

    Returns:
        str: Emotion of the biggest face in the picture or None if no face was found
    """
    try:
        objs = DF(img_path=img, actions=["age", "gender", "race", "emotion"])
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


def run_mesh_detection_model(img: np.ndarray) -> np.ndarray:  # ADD: return type hint
    """Run the face mesh detection model that uses MediaPipe as a backend for one face in the image. If multiple faces are present in the image, the one with highest confidence is considered.

    Args:
        img (np.ndarray): Input image

    Returns:
        np.ndarray: Normalized list or facemesh points of the highest confedince face in the given image
    """
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        results = face_mesh.process(rgb_img)
        if results.multi_face_landmarks:
            normalized_mesh_point_list = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )
            return normalized_mesh_point_list
        else:
            raise ValueError(
                "No face could be detected. Please confirm that the picture is a face photo."
            )


def get_face_mesh(url: str) -> np.ndarray:
    """Run facemesh detection on a face image from a URL, return normalized facemesh point list
    Args:
        url (str): Image URL to be processed.

    Raises:
        IOError: In case URL is not valid or doesn't contain an image

    Returns:
        np.ndarray: Normalized list or facemesh points of the highest confedince face in the given image
    """
    success, img = get_image(url)
    if success:
        mesh_point_list = run_mesh_detection_model(img)
        return mesh_point_list
    else:
        raise IOError("URL not valid or not an image", "URL is: {url}")


def get_skin_hue(url: str, point_num: int) -> np.uint8:
    """Gets skin hue from a face image given a URL and the desired facemesh point to investigate.

    Args:
        url (str): Image URL to be processed
        point_num (int): Facemesh point to get its skin hue

    Raises:
        ValueError: In case a wrong facemesh point number was given
        IOError: If give URL is not a valid URL or is not an image URL.

    Returns:
        np.uint8: Skin hue value of the desired facemesh point
    """
    if not (point_num >= 0 and point_num <= 477):
        raise ValueError(
            "Wrong facemesh point, facemesh points are integers between 0 and 477."
        )
    success, img = get_image(url)
    if success:
        normalized_mesh_point_list = run_mesh_detection_model(img)
        hue = get_hue(img, normalized_mesh_point_list[point_num])
        return hue
    else:
        raise IOError("URL not valid or not an image", "URL is: {url}")


def get_eyelid_distance(url: str) -> list:
    """Given a URL of a face image, return the vertical and horizontal distance of the eyelids of each eye.

    Args:
        url (str): Input image URL.

    Returns:
        list: A list containing tuples for Left eye vertical, Left eye horizontal, Right eye vertical, and Right eye horizontal distances.
    """
    normalized_mesh_point_list = get_face_mesh(url)
    return calculate_distance(normalized_mesh_point_list)


def get_iris_radius(url: str) -> list:
    """Given a URL of a face image, return the center and radius of the left and right iris.

    Args:
        url (str): Input Image URL.

    Returns:
        list: A list containing tuples for Left eye center, Left eye radius, Right eye center, and Right eye radius.
    """
    normalized_mesh_point_list = get_face_mesh(url)
    return calculate_iris_radius(normalized_mesh_point_list)
