import hmac, io, json, os, threading, time
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

meta = json.load(open("champion.json"))
THRESHOLD = float(meta.get("threshold", 0.5))
MODEL_NAME = str(meta.get("model_name", "visionguard"))
MODEL_VERSION = str(meta.get("version", "unknown"))
API_KEY = os.environ.get("API_KEY", "")
ALLOWED = {"JPEG", "PNG", "WEBP"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

interp = Interpreter(model_path="model.tflite")   # loaded ONCE at start-up
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
SIZE = (int(inp["shape"][2]), int(inp["shape"][1]))   # (width, height)
lock = threading.Lock()


def err(msg, code):
    return jsonify({"error": msg}), code


@app.errorhandler(413)
def too_big(e):
    return err("file too large (max 5 MB)", 413)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME,
                    "version": MODEL_VERSION, "threshold": THRESHOLD})


@app.post("/predict")
def predict():
    key = request.headers.get("X-API-Key", "")
    if not API_KEY or not hmac.compare_digest(key, API_KEY):
        return err("invalid or missing API key", 401)
    f = request.files.get("file")
    if f is None:
        return err("send the image in form field 'file'", 400)
    data = f.read()
    try:
        Image.open(io.BytesIO(data)).verify()
        im = Image.open(io.BytesIO(data))
        fmt, (w, h) = im.format, im.size
        im = im.convert("RGB")
    except Exception:
        return err("file is not a decodable image", 400)
    if fmt not in ALLOWED:
        return err(f"unsupported type {fmt}", 415)
    t0 = time.perf_counter()
    x = np.asarray(im.resize(SIZE, Image.BILINEAR), dtype=np.float32)[None]
    with lock:
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        p = float(interp.get_tensor(out["index"])[0, 0])
    is_ai = p >= THRESHOLD
    return jsonify({"label": "AI-generated" if is_ai else "Real",
                    "confidence": p if is_ai else 1 - p,
                    "prob_ai": p, "threshold": THRESHOLD,
                    "model_name": MODEL_NAME, "model_version": MODEL_VERSION,
                    "width": w, "height": h, "file_size": len(data),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})