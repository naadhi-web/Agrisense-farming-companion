from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template_string, request
from PIL import Image


CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "trained_plant_disease_model.keras"
GUIDE_PATH = APP_DIR.parent / "DISEASE-GUIDE.md"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

_model: Optional[tf.keras.Model] = None
_guide_by_label: Optional[Dict[str, str]] = None

HOME_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AgriSense Plant Disease Server</title>
  <style>
    :root { color-scheme: light; }
    body { font-family: "Segoe UI", sans-serif; background: #f4f7f2; margin: 0; padding: 24px; color: #1f2937; }
    .card { max-width: 760px; margin: 0 auto; background: #fff; border-radius: 14px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,.08); }
    h1 { margin: 0 0 8px; color: #2f7d32; }
    p { margin: 0 0 16px; }
    form { display: grid; gap: 12px; }
    input[type=file] { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; }
    button { background: #2f7d32; color: #fff; border: 0; border-radius: 8px; padding: 10px 14px; font-weight: 600; cursor: pointer; width: fit-content; }
    button:disabled { opacity: .7; cursor: not-allowed; }
    .small { color: #4b5563; font-size: 14px; }
    .result { margin-top: 16px; background: #f8faf8; border: 1px solid #dbe7d6; border-radius: 8px; padding: 12px; white-space: pre-wrap; }
    .error { color: #b42318; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Plant Disease Prediction</h1>
    <p>Upload a leaf image to get top disease predictions from the trained AgriSens model.</p>
    <form id="predict-form">
      <input id="image-input" name="image" type="file" accept="image/*" required>
      <button id="submit-btn" type="submit">Predict</button>
    </form>
    <p class="small">API endpoints: <code>GET /health</code>, <code>GET /classes</code>, <code>POST /predict</code></p>
    <div id="result" class="result">No prediction yet.</div>
  </div>
  <script>
    const form = document.getElementById("predict-form");
    const imageInput = document.getElementById("image-input");
    const submitBtn = document.getElementById("submit-btn");
    const resultEl = document.getElementById("result");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!imageInput.files.length) {
        resultEl.textContent = "Please choose an image first.";
        resultEl.classList.add("error");
        return;
      }

      const formData = new FormData();
      formData.append("image", imageInput.files[0]);

      submitBtn.disabled = true;
      resultEl.classList.remove("error");
      resultEl.textContent = "Predicting...";

      try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Prediction failed");
        }

        const top = data.top_predictions || [];
        const topLines = top.map((item, idx) => `${idx + 1}. ${item.label} (${item.confidence_percent}%)`);
        const guidance = data.guidance ? `\\n\\nGuidance:\\n${data.guidance}` : "";
        resultEl.textContent = `Top prediction: ${data.prediction} (${data.confidence_percent}%)\\n\\nTop 3:\\n${topLines.join("\\n")}${guidance}`;
      } catch (error) {
        resultEl.textContent = `Error: ${error.message}`;
        resultEl.classList.add("error");
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip().lower()


def load_model() -> tf.keras.Model:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(str(MODEL_PATH))
    return _model


def load_disease_guide() -> dict[str, str]:
    global _guide_by_label
    if _guide_by_label is not None:
        return _guide_by_label

    if not GUIDE_PATH.exists():
        _guide_by_label = {}
        return _guide_by_label

    text = GUIDE_PATH.read_text(encoding="utf-8")
    sections = re.split(r"\n###\s+", text)
    mapping: Dict[str, str] = {}
    for section in sections:
        section = section.strip()
        if not section:
            continue
        first_line, *rest = section.splitlines()
        match = re.match(r"^\d+\.\s*(.*)$", first_line)
        label = match.group(1).strip() if match else first_line.strip()
        mapping[label] = "\n".join(rest).strip()

    _guide_by_label = mapping
    return _guide_by_label


def preprocess_image(file_storage: Any) -> np.ndarray:
    image = Image.open(file_storage).convert("RGB")
    image = image.resize((128, 128))
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    return np.expand_dims(image_arr, axis=0)


def prediction_response(predictions: np.ndarray) -> dict[str, Any]:
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_predictions = [
        {
            "label": CLASS_NAMES[index],
            "confidence": float(predictions[index]),
            "confidence_percent": round(float(predictions[index]) * 100.0, 2),
        }
        for index in top_indices
    ]

    top_label = top_predictions[0]["label"]
    guide = load_disease_guide()
    guidance = guide.get(top_label)
    if guidance is None:
        # Fallback for minor formatting differences.
        normalized_map = {_normalize_label(key): value for key, value in guide.items()}
        guidance = normalized_map.get(_normalize_label(top_label))

    return {
        "prediction": top_label,
        "confidence": top_predictions[0]["confidence"],
        "confidence_percent": top_predictions[0]["confidence_percent"],
        "top_predictions": top_predictions,
        "guidance": guidance,
    }


@app.after_request
def add_cors_headers(response):  # type: ignore[no-untyped-def]
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/health", methods=["GET"])
def health() -> Any:
    model_loaded = _model is not None
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model_loaded,
            "model_path": str(MODEL_PATH),
            "classes": len(CLASS_NAMES),
        }
    )


@app.route("/classes", methods=["GET"])
def classes() -> Any:
    return jsonify({"classes": CLASS_NAMES})


@app.route("/", methods=["GET"])
def home() -> Any:
    return render_template_string(HOME_PAGE)


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded. Use form field name 'image'."}), 400

    try:
        model = load_model()
        image_arr = preprocess_image(file)
        preds = model.predict(image_arr, verbose=0)[0]
        return jsonify(prediction_response(preds))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover - runtime safety net
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.errorhandler(413)
def payload_too_large(_error) -> Any:  # type: ignore[no-untyped-def]
    return jsonify({"error": "Image too large. Max allowed size is 10MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
