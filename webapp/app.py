from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from webapp.detect import predict

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/detect", methods=["POST"])
def detect():
    try:
        file = request.files["file"]

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        print("FILE SAVED:", path)

        result, output_path = predict(path)

        print("OUTPUT PATH:", output_path)

        if output_path is None:
            return jsonify({
                "result": result,
                "image": None
            })

        filename = os.path.basename(output_path)

        return jsonify({
            "result": result,
            "image": "/uploads/" + filename
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)