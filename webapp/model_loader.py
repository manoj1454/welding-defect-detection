import torch
import torchvision
import os
import gdown

# =========================
# DEVICE
# =========================
device = torch.device("cpu")

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "weld_final.pth")

# =========================
# GOOGLE DRIVE FILE ID
# =========================
FILE_ID = "13bKS6a9M6qua3ocr_7W4FAJA7xXVf2RH"

MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"


# =========================
# DOWNLOAD MODEL
# =========================
def download_model():
    print("Downloading from:", MODEL_URL)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("Downloading model...")

    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    print("Model downloaded successfully!")


# =========================
# LOAD MODEL
# =========================
def load_model():

    # 🔥 IMPORTANT FIX: check BEFORE loading
    if not os.path.exists(MODEL_PATH):
        download_model()

    num_classes = 4

    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None)

    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels

    model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
        in_channels,
        num_anchors,
        num_classes
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.to(device)
    model.eval()

    return model