"""
Model Loader - Lazy loading and caching of ML models
"""

import os
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Default model paths (relative to this file's directory)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "weights")


@lru_cache(maxsize=1)
def get_dlib_predictor():
    """
    Get dlib shape predictor for 68-point facial landmarks.
    
    Model file: shape_predictor_68_face_landmarks.dat
    Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    
    Returns:
        dlib.shape_predictor instance
    """
    import dlib
    
    # Try multiple locations
    possible_paths = [
        os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                     "Artificial-Intelligence-based-Online-Exam-Proctoring-System",
                     "shape_predictor_model", "shape_predictor_68_face_landmarks.dat"),
        "shape_predictor_68_face_landmarks.dat"  # Current directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Loading dlib predictor from: {path}")
            return dlib.shape_predictor(path)
    
    # If not found, raise helpful error
    raise FileNotFoundError(
        f"shape_predictor_68_face_landmarks.dat not found. "
        f"Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
        f"and place in {MODELS_DIR}"
    )


@lru_cache(maxsize=1)
def get_yolo_model():
    """
    Get YOLO model for prohibited object detection.
    
    Model file: OEP_YOLOv11n.pt (from AutoOEP)
    
    Returns:
        YOLO model instance
    """
    from ultralytics import YOLO
    
    # Try multiple locations
    possible_paths = [
        os.path.join(MODELS_DIR, "OEP_YOLOv11n.pt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                     "AutoOEP", "Models", "OEP_YOLOv11n.pt"),
        "OEP_YOLOv11n.pt"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Loading YOLO model from: {path}")
            model = YOLO(path)
            return model
    
    # If custom model not found, use a default YOLO model
    logger.warning("Custom YOLO model not found, using yolov8n as fallback")
    return YOLO("yolov8n.pt")


@lru_cache(maxsize=1)
def get_mediapipe_hands():
    """
    Get MediaPipe Hands instance.
    
    Returns:
        Tuple of (mpHands module, hands instance, drawing utils)
    """
    import mediapipe as mp
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    logger.info("MediaPipe Hands initialized")
    
    return {
        "mpHands": mp_hands,
        "hands": hands,
        "mpdraw": mp_draw
    }


def get_face_landmarker_path() -> Optional[str]:
    """
    Get path to MediaPipe face landmarker task file.
    
    Returns:
        Path to face_landmarker.task or None if not found
    """
    possible_paths = [
        os.path.join(MODELS_DIR, "face_landmarker.task"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                     "AutoOEP", "Models", "face_landmarker.task"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    logger.warning("face_landmarker.task not found")
    return None


def check_models() -> dict:
    """
    Check which models are available.
    
    Returns:
        Dict with model status
    """
    status = {
        "dlib_predictor": False,
        "yolo_model": False,
        "mediapipe": False,
        "face_landmarker": False
    }
    
    # Check dlib predictor
    for path in [
        os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat"),
        "shape_predictor_68_face_landmarks.dat"
    ]:
        if os.path.exists(path):
            status["dlib_predictor"] = True
            break
    
    # Check YOLO model
    for path in [
        os.path.join(MODELS_DIR, "OEP_YOLOv11n.pt"),
        "OEP_YOLOv11n.pt"
    ]:
        if os.path.exists(path):
            status["yolo_model"] = True
            break
    
    # Check MediaPipe
    try:
        import mediapipe
        status["mediapipe"] = True
    except ImportError:
        pass
    
    # Check face landmarker
    if get_face_landmarker_path():
        status["face_landmarker"] = True
    
    return status
