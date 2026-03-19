from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from detect import predict

# =========================
# INIT APP
# =========================
app = Flask(__name__)

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# SERVE UPLOADED FILES
# =========================
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# =========================
# DETECTION API
# =========================
@app.route("/detect", methods=["POST"])
def detect():

    # Check file exists
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save input image
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # Run model
    result, output_path = predict(input_path)

    # Extract filename
    output_filename = os.path.basename(output_path)

    # Return JSON response
    return jsonify({
        "result": result,
        "image_url": f"/uploads/{output_filename}"
    })


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)