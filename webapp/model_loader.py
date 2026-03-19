import torch
import torchvision
import os

# =========================
# DEVICE
# =========================

device = torch.device("cpu")  # Render = CPU only


# =========================
# PATH FIX (IMPORTANT)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "weld_final.pth")


# =========================
# LOAD MODEL FUNCTION
# =========================

def load_model():

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