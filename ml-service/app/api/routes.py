from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
import mediapipe as mp
from app.services.hand_landmarker import get_hand_bbox
from app.services.preprocessing import preprocess_hand_image

router = APIRouter()

@router.post("/predict", response_class=JSONResponse)
async def predict(
    request: Request,
    frame: UploadFile = File(...)
):
    # 1️⃣ Read image
    image_bytes = await frame.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2️⃣ MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    hand_landmarker = request.app.state.hand_landmarker
    result = hand_landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return {
            "hand_detected": False,
            "char": "",
            "confidence": 0.0
        }

    # 3️⃣ FIXED bbox call (no missing args)
    h, w, _ = img.shape
    bbox = get_hand_bbox(result.hand_landmarks[0], image_width=w, image_height=h)

    x1, y1, x2, y2 = bbox
    hand_crop = img[y1:y2, x1:x2]

    hand_tensor = preprocess_hand_image(hand_crop)

    # 4️⃣ Tensor shape fix (ONLY 4D allowed)
    input_tensor = hand_tensor  # already [1, 3, 224, 224]

    if input_tensor.ndim != 4:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid input tensor shape: {input_tensor.shape}"
        )

    model = request.app.state.model
    device = request.app.state.device

    # 5️⃣ Inference
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

    predicted_char = chr(ord("A") + idx.item())

    return {
        "hand_detected": True,
        "char": predicted_char,
        "confidence": float(conf.item())
    }
