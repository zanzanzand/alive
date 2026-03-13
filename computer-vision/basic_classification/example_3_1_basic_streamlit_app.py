from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Input images are 128x128, after two poolings -> 32x32
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model_from_bytes(model_bytes: bytes, device: torch.device) -> nn.Module:
    model = SimpleCNN().to(device)
    state = torch.load(io.BytesIO(model_bytes), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
        ]
    )
    return transform(image_rgb)


def decode_image(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    img_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode image.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr, image_rgb


st.set_page_config(page_title="Cats vs Dogs", page_icon="🐾")

st.title("Cats vs Dogs")
st.write("Upload an image and classify it using a saved model.")

model_file = st.file_uploader("Upload model (.pth)", type=["pth"])
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if model_file is not None and uploaded_file is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_bytes(model_file.read(), device)

    file_bytes = uploaded_file.read()
    image_bgr, image_rgb = decode_image(file_bytes)
    st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

    input_tensor = preprocess_image(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    class_names = ["cat", "dog"]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]

    st.write(f"Prediction: **{pred_label}**")
    st.write(f"Probabilities: cat={probs[0]:.3f}, dog={probs[1]:.3f}")
else:
    st.info("Upload a model and an image to classify.")
