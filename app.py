import io
import os

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from mkunet_network import MK_UNet

st.set_page_config(page_title="MK-UNet Segmentation", layout="wide")

# ---------------- SETTINGS (change only if needed) ----------------
WEIGHTS_PATH = "models/best.pth"
IMG_SIZE = 352                      # same size used in training
MEAN = (0.485, 0.456, 0.406)        # see note below about these
STD = (0.229, 0.224, 0.225)
VARIANTS = {
    "MK_UNet_T": [4, 8, 16, 24, 32],
    "MK_UNet_S": [8, 16, 32, 48, 80],
    "MK_UNet":   [16, 32, 64, 96, 160],
    "MK_UNet_M": [32, 64, 128, 192, 320],
    "MK_UNet_L": [64, 128, 256, 384, 512],
}
# ------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model...")
def load_model(weights_bytes, variant):
    model = MK_UNet(num_classes=1, in_channels=3, channels=VARIANTS[variant])
    state = torch.load(io.BytesIO(weights_bytes), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    model.eval()
    return model, result.missing_keys


def segment(model, image, threshold):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        out = out[0] if isinstance(out, (list, tuple)) else out
        out = F.interpolate(out, size=(image.height, image.width),
                            mode="bilinear", align_corners=False)
        out = out.sigmoid().squeeze()
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)  # same as your test script
    return (out >= threshold).numpy().astype(np.uint8)


def make_overlay(image, mask, alpha=0.5):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    red = np.array([255, 60, 60], dtype=np.float32)
    blend = np.where(mask[..., None] > 0, (1 - alpha) * base + alpha * red, base)
    return Image.fromarray(blend.astype(np.uint8))


# ----------------------------- PAGE -------------------------------
st.title("MK-UNet Image Segmentation")

with st.sidebar:
    variant = st.selectbox("Model size", list(VARIANTS), index=2)
    threshold = st.slider("Threshold", 0.05, 0.95, 0.5, 0.05)

# load weights from the models folder, or ask for an upload if not found
if os.path.exists(WEIGHTS_PATH):
    with open(WEIGHTS_PATH, "rb") as f:
        weights_bytes = f.read()
else:
    st.warning("models/best.pth not found. Upload your .pth file below.")
    up = st.file_uploader("Upload weights (.pth)", type=["pth", "pt"])
    if up is None:
        st.stop()
    weights_bytes = up.getvalue()

model, missing = load_model(weights_bytes, variant)
if missing:
    st.error("The weights don't match this model size. Try another option "
             "in 'Model size' in the sidebar.")
    st.stop()

image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if image_file:
    image = Image.open(image_file).convert("RGB")
    mask = segment(model, image, threshold)

    c1, c2, c3 = st.columns(3)
    c1.image(image, caption="Input", use_container_width=True)
    c2.image(mask * 255, caption="Predicted mask", use_container_width=True)
    c3.image(make_overlay(image, mask), caption="Overlay", use_container_width=True)

    buf = io.BytesIO()
    Image.fromarray(mask * 255).save(buf, format="PNG")
    st.download_button("Download mask", buf.getvalue(), "mask.png", "image/png")