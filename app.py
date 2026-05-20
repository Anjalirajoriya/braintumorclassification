from flask import Flask, render_template, request, redirect, url_for
from model_utils import predict_image, CLASS_NAMES
import os, uuid

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    label, confidence, probs, stats = predict_image(filepath)

    # Not a brain MRI
    if label == "not_mri":
        return render_template("result.html",
            label="Not a Brain MRI",
            confidence=0,
            top3=[],
            image_url=filepath,
            error="⚠️ This image does not appear to be a brain MRI scan. Please upload a valid MRI image.")

    # Model is uncertain
    if label == "uncertain":
        return render_template("result.html",
            label="Uncertain",
            confidence=round(confidence * 100, 2),
            top3=[],
            image_url=filepath,
            error="⚠️ Model is not confident enough to make a prediction. Please try a clearer MRI image.")

    top3 = sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda x: -x[1])[:3]
    top3 = [(n.replace("_", " ").title(), round(s * 100, 2)) for n, s in top3]

    return render_template("result.html",
        label=label.replace("_", " ").title(),
        confidence=round(confidence * 100, 2),
        top3=top3,
        image_url=filepath,
        error=None)

if __name__ == "__main__":
    app.run(debug=True)