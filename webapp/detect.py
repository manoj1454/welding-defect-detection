import cv2
import torch
import os

from webapp.model_loader import load_model

device = torch.device("cpu")

model = load_model()
model.to(device)
model.eval()

CLASS_NAMES = [
    "background",
    "defect_1",
    "defect_2",
    "defect_3"
]


def predict(image_path):
    print("PREDICT CALLED")

    try:
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError("Image not loaded")

        original = img.copy()

        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ✅ CORRECT conversion
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)

        boxes = outputs[0]["boxes"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()

        threshold = 0.3
        detected = False

        for box, score, label in zip(boxes, scores, labels):
            if score < threshold:
                continue

            detected = True

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 0, 255), 2)

            label = int(label)
            if label < len(CLASS_NAMES):
                text = f"{CLASS_NAMES[label]} {score:.2f}"

                cv2.putText(
                    original,
                    text,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # ✅ SAVE OUTPUT
        base, _ = os.path.splitext(image_path)
        output_path = base + "_result.jpg"

        cv2.imwrite(output_path, original)

        print("Image saved:", output_path)

        result = "Bad Weld (Defect Detected)" if detected else "Good Weld"

        print("DETECTION DONE:", result)

        return result, output_path

    except Exception as e:
        print("ERROR:", str(e))
        return "Error occurred", None