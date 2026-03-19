import cv2
import torch
import os

from webapp.model_loader import load_model

# =========================
# DEVICE
# =========================
device = torch.device("cpu")

# =========================
# LOAD MODEL (ONCE)
# =========================
model = load_model()
model.to(device)
model.eval()

# =========================
# CLASS NAMES
# =========================
CLASS_NAMES = [
    "background",
    "defect_1",
    "defect_2",
    "defect_3"
]

# =========================
# PREDICT FUNCTION
# =========================
def predict(image_path):
    try:
        print("Prediction started")

        img = cv2.imread(image_path)

        if img is None:
            raise ValueError("Image not loaded properly")

        original = img.copy()

        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ✅ FIXED PART
        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)

        print("Model inference done")

        boxes = outputs[0]["boxes"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()

        print("Processing outputs")

        threshold = 0.3
        detected = False

        for box,score,label in zip(boxes,scores,labels):
            if score < threshold:
                continue

            detected = True

            x1,y1,x2,y2 = map(int,box)
            cv2.rectangle(original,(x1,y1),(x2,y2),(0,0,255),2)

        output_filename = os.path.basename(image_path).replace(".jpg","_result.jpg")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(BASE_DIR,"uploads",output_filename)

        cv2.imwrite(output_path, original)

        print("Image saved:", output_path)

        result = "Bad Weld (Defect Detected)" if detected else "Good Weld"

        return result, output_filename

    except Exception as e:
        print("ERROR:", str(e))
        return "Error occurred", None

    # =========================
    # DRAW BOXES
    # =========================
    for box, score, label in zip(boxes, scores, labels):

        if score < threshold:
            continue

        detected = True

        x1, y1, x2, y2 = map(int, box)

        # Safety check
        label = int(label)
        if label >= len(CLASS_NAMES):
            continue

        class_name = CLASS_NAMES[label]

        text = f"{class_name} ({score:.2f})"

        # Draw rectangle
        cv2.rectangle(original, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw label
        cv2.putText(
            original,
            text,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # =========================
    # SAVE OUTPUT IMAGE (FIXED)
    # =========================
    base, ext = os.path.splitext(image_path)
    output_path = base + "_result.jpg"

    cv2.imwrite(output_path, original)

    # =========================
    # RESULT TEXT
    # =========================
    if detected:
        result = "Bad Weld (Defect Detected)"
    else:
        result = "Good Weld"

    return result, output_path