import cv2
import torch
import numpy as np

def preprocess_hand_image(hand_img):
    # Resize
    hand_img = cv2.resize(hand_img, (224, 224))

    # BGR → RGB
    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1]
    hand_img = hand_img / 255.0

    # ImageNet normalization (used by ResNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    hand_img = (hand_img - mean) / std

    # HWC → CHW
    hand_img = np.transpose(hand_img, (2, 0, 1))

    # To tensor
    tensor = torch.tensor(hand_img, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)  # (1, 3, 224, 224)

    return tensor
