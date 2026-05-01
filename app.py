import csv
import io
from collections import deque
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, Response, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
DEFAULT_MODEL_PATHS = (
    BASE_DIR / "runs_cls" / "pothole_cls" / "weights" / "best.pt",
    BASE_DIR / "runs" / "classify" / "runs_cls" / "pothole_cls" / "weights" / "best.pt",
)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_HISTORY = 20

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
history = deque(maxlen=MAX_HISTORY)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def model_ready() -> bool:
    return resolve_model_path() is not None


def resolve_model_path() -> Path | None:
    for path in DEFAULT_MODEL_PATHS:
        if path.exists():
            return path

    # Fallback: pick the most recently modified trained best.pt
    candidates = []
    candidates.extend((BASE_DIR / "runs_cls").glob("*/weights/best.pt"))
    candidates.extend((BASE_DIR / "runs" / "classify" / "runs_cls").glob("*/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def clamp_threshold(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, value))


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None
    threshold = 0.5

    if request.method == "POST":
        file = request.files.get("image")
        threshold = clamp_threshold(request.form.get("threshold", "0.5"))

        if file is None or file.filename == "":
            error = "Please select an image file."
        elif not allowed_file(file.filename):
            error = "Unsupported file type. Use jpg, jpeg, png, or webp."
        elif not model_ready():
            error = (
                "Model not found in runs_cls. Please run training first "
                "(python train.py)."
            )
        else:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = secure_filename(file.filename)
            unique_name = f"{uuid4().hex}_{safe_name}"
            file_path = UPLOAD_DIR / unique_name
            file.save(file_path)

            model_path = resolve_model_path()
            if model_path is None:
                error = (
                    "Model not found in runs_cls. Please run training first "
                    "(python train.py)."
                )
                return render_template(
                    "index.html",
                    result=None,
                    error=error,
                    history=list(history),
                    threshold=threshold,
                )

            model = YOLO(str(model_path))
            pred = model.predict(source=str(file_path), conf=0.25, verbose=False)[0]

            probs = pred.probs
            names = {v.lower(): k for k, v in pred.names.items()}
            pothole_id = names.get("pothole")
            pothole_prob = float(probs.data[pothole_id]) if pothole_id is not None else 0.0

            label = (
                "Pothole detected"
                if pothole_prob >= threshold
                else "No pothole detected"
            )
            final_verdict = "Pothole" if pothole_prob >= threshold else "Not Pothole"

            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "label": label,
                "final_verdict": final_verdict,
                "pothole_probability": f"{pothole_prob * 100:.2f}%",
                "pothole_probability_value": pothole_prob,
                "threshold": f"{threshold * 100:.0f}%",
                "threshold_value": threshold,
                "filename": file_path.name,
                "image_url": f"/static/uploads/{file_path.name}",
            }
            history.appendleft(result)

    return render_template(
        "index.html",
        result=result,
        error=error,
        history=list(history),
        threshold=threshold,
    )


@app.route("/download-history.csv", methods=["GET"])
def download_history_csv() -> Response:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "timestamp",
            "filename",
            "decision",
            "pothole_probability_percent",
            "threshold_percent",
        ]
    )
    for item in history:
        writer.writerow(
            [
                item.get("timestamp", ""),
                item.get("filename", ""),
                item.get("label", ""),
                f"{item.get('pothole_probability_value', 0.0) * 100:.2f}",
                f"{item.get('threshold_value', 0.5) * 100:.0f}",
            ]
        )

    csv_data = output.getvalue()
    output.close()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=pothole_results.csv"},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
