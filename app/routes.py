import base64
import io
import logging
import os
import time

from flask import Blueprint, current_app, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

bp = Blueprint("main", __name__)
log = logging.getLogger("visionguard")

ALLOWED_EXT = {".png", ".jpg", ".jpeg"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_PIXELS = 40_000_000


class ValidationError(Exception):
    pass


def wants_json():
    return request.headers.get("X-Requested-With") == "fetch"


def fail(code, title, message):
    """Friendly error: JSON for the page's JavaScript, an HTML page otherwise."""
    if wants_json():
        return jsonify({"error": message, "title": title}), code
    return render_template("error.html", code=code, title=title, message=message), code


def validate_upload(file):
    if file is None or not file.filename:
        raise ValidationError("No file was selected. Please choose an image.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise ValidationError("Unsupported file type. Please upload a JPG or PNG image.")
    safe_name = secure_filename(file.filename) or f"upload{ext}"   # sanitize the name
    data = file.read()                                             # stays in memory only
    if not data:
        raise ValidationError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
            im.verify()
    except (OSError, SyntaxError, ValueError):
        raise ValidationError("This file looks like an image but could not be decoded. "
                              "It may be corrupt or not really an image.") from None
    if fmt not in ALLOWED_FORMATS:
        raise ValidationError("The file content is not a JPG or PNG image.")
    if w * h > MAX_PIXELS:
        raise ValidationError("The image has too many pixels. Please upload a smaller one.")
    return safe_name, data, w, h


def thumbnail_uri(data):
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((480, 480))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/predict", methods=["POST"])
def analyze():
    try:
        name, data, w, h = validate_upload(request.files.get("image"))
    except ValidationError as e:
        return fail(400, "We couldn't use that file", str(e))

    t0 = time.perf_counter()
    try:
        res = current_app.extensions["predictor"].predict(data)
    except Exception:
        log.exception("Inference failed")
        return fail(500, "Analysis failed", "Something went wrong while analysing the image.")
    ms = (time.perf_counter() - t0) * 1000

    cfg = current_app.config
    saved = True
    try:
        current_app.extensions["db"].add_prediction(
            filename=name[:255], label=res["label"], confidence=res["confidence"],
            prob_ai=res["prob_ai"], threshold=res["threshold"],
            model_name=res.get("model_name") or cfg["MODEL_NAME"],
            model_version=str(res.get("model_version") or cfg["MODEL_VERSION"]),
            width=w, height=h, file_size=len(data), latency_ms=ms)
    except Exception:
        log.exception("Could not write to the database")
        saved = False

    if wants_json():
        return jsonify({"label": res["label"], "confidence": res["confidence"],
                        "prob_ai": res["prob_ai"], "threshold": res["threshold"],
                        "model": cfg["MODEL_NAME"], "model_version": str(cfg["MODEL_VERSION"]),
                        "filename": name, "width": w, "height": h, "file_size": len(data),
                        "latency_ms": round(ms, 1), "saved": saved})
    return render_template("result.html", res=res, name=name, w=w, h=h, saved=saved,
                           preview=thumbnail_uri(data), model=cfg["MODEL_NAME"],
                           version=cfg["MODEL_VERSION"])


@bp.route("/history")
def history():
    rows = current_app.extensions["db"].recent(100)
    n = len(rows)
    ai = sum(1 for r in rows if r.label == "AI-generated")
    stats = {"total": n, "ai": ai, "real": n - ai,
             "avg": (sum(r.confidence for r in rows) / n * 100) if n else 0}
    return render_template("history.html", rows=rows, stats=stats)


@bp.route("/health")
def health():
    db_ok = current_app.extensions["db"].ping()
    cfg = current_app.config
    body = {"status": "ok" if db_ok else "degraded", "model_loaded": True,
            "model": cfg["MODEL_NAME"], "model_version": str(cfg["MODEL_VERSION"]),
            "threshold": cfg["THRESHOLD"], "database": "ok" if db_ok else "error"}
    return jsonify(body), (200 if db_ok else 503)


@bp.app_errorhandler(404)
def not_found(e):
    return fail(404, "Page not found", "That page doesn't exist.")


@bp.app_errorhandler(405)
def bad_method(e):
    return fail(405, "Not allowed", "That action isn't allowed on this page.")


@bp.app_errorhandler(413)
def too_large(e):
    mb = current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return fail(413, "File too large", f"The file is larger than the {mb} MB limit.")


@bp.app_errorhandler(500)
def server_error(e):
    return fail(500, "Server error", "Something went wrong on our side. Please try again.")