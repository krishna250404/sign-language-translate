from fastapi import FastAPI
from app.api.routes import router
from app.services.hand_landmarker import load_hand_landmarker
from app.model.model import SignLanguageModel
import torch

app = FastAPI(title="Sign Language ML Service")

@app.on_event("startup")
def load_resources():
    # Hand landmarker
    app.state.hand_landmarker = load_hand_landmarker()

    # ML model
    device = torch.device("cpu")
    model = SignLanguageModel(num_classes=26)

    checkpoint = torch.load(
        "app/model/best_model.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    app.state.model = model
    app.state.device = device

app.include_router(router)

@app.get("/")
def health():
    return {"status": "ok"}
