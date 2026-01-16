import mediapipe as mp
from app.core.config import HAND_LANDMARKER_PATH
import mediapipe as mp
from pathlib import Path

def load_hand_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Resolve path safely
    model_path = Path(__file__).resolve().parent.parent / "model" / "hand_landmarker.task"

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )

    return HandLandmarker.create_from_options(options)

def extract_landmark_vector(hand_landmarks):
    vector = []
    for lm in hand_landmarks:
        vector.extend([lm.x, lm.y, lm.z])
    return vector

def get_hand_bbox(landmarks, image_width, image_height, margin=20):
    xs = [int(lm.x * image_width) for lm in landmarks]
    ys = [int(lm.y * image_height) for lm in landmarks]

    x1 = max(min(xs) - margin, 0)
    y1 = max(min(ys) - margin, 0)
    x2 = min(max(xs) + margin, image_width)
    y2 = min(max(ys) + margin, image_height)

    return x1, y1, x2, y2


