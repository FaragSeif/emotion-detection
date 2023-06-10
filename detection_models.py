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


def run_mesh_detection_model(img: np.ndarray):  # ADD: return type hint
    """Run the face mesh detection model that uses MediaPipe as a backend for one face in the image. If multiple faces are present in the image, the one with highest confidence is considered.

    Args:
        img (np.ndarray): Input image

    Returns:
        : Face mesh of the highest confedince face in the given image
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


def get_face_mesh(url: str) -> str:
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
        mesh_point_list = run_mesh_detection_model(img)
        return mesh_point_list
    else:
        raise IOError("URL not valid or not an image", "URL is: {url}")


def get_skin_hue(url, point_num):
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


def get_eyelid_distance(url):
    normalized_mesh_point_list = get_face_mesh(url)
    return calculate_distance(normalized_mesh_point_list)


def get_iris_radius(url):
    normalized_mesh_point_list = get_face_mesh(url)
    return calculate_iris_radius(normalized_mesh_point_list)
