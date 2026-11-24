"""Flask web app: upload an apple photo, get a predicted disease and treatment advice.

The model has its preprocessing built in (see src/models.py), so the app only
resizes the image. It must NOT call preprocess_input itself.

Run:  python app.py   (expects models/resnet50.keras; see README "Get the model")
"""
import io
import os

import keras
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

import src.models  # noqa: F401  (registers the custom preprocessing layer)
from src.config import CLASS_NAMES, IMG_SIZE, MODELS_DIR

MODEL_PATH = os.environ.get("MODEL_PATH", str(MODELS_DIR / "resnet50.keras"))

TREATMENT = {
    "Blotch_Apple": "Apply a suitable fungicide spray containing captan or mancozeb. "
                    "Prune infected branches and destroy fallen leaves.",
    "Normal_Apple": "No specific treatment needed. Maintain good cultural practices, "
                    "such as proper watering and fertilization.",
    "Rot_Apple": "Discard infected apples. Maintain proper storage conditions with good "
                 "ventilation and controlled humidity.",
    "Scab_Apple": "Apply a suitable fungicide spray containing myclobutanil or sulfur. "
                  "Prune and destroy infected plant parts.",
}
PESTICIDE = {
    "Blotch_Apple": "Use fungicides containing captan or mancozeb at recommended doses.",
    "Normal_Apple": "No specific pesticide needed.",
    "Rot_Apple": "Maintain proper storage conditions, including temperature and humidity "
                 "control. No specific pesticide recommended.",
    "Scab_Apple": "Use fungicides containing myclobutanil or sulfur as per label instructions.",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
model = keras.models.load_model(MODEL_PATH)


def model_predict(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    batch = np.asarray(img, dtype="float32")[None]  # raw 0-255 RGB, as in training
    probs = model.predict(batch, verbose=0)[0]
    label = CLASS_NAMES[int(np.argmax(probs))]
    return {
        "Class": label,
        "Confidence": f"{probs.max():.0%}",
        "Treatment": TREATMENT[label],
        "Pesticide": PESTICIDE[label],
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/aboutus", methods=["GET"])
def aboutus():
    return render_template("about.html")


@app.route("/check_disease", methods=["GET"])
def check_disease():
    return render_template("upload_file.html")


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method != "POST" or not request.files.get("file"):
        return render_template("upload_file.html")
    preds = model_predict(request.files["file"].read())
    return render_template("result.html", data=preds)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
