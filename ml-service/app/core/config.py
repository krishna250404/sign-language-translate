from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

PYTORCH_MODEL_PATH = MODEL_DIR / "best_model.pth"
HAND_LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"
