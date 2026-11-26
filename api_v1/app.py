"""VisionGuard model API for an ONNX image classifier from the Hugging Face Hub.

Same contract as the TFLite API in api/: GET /health and POST /predict
(X-API-Key header, multipart field "file").
The model files are read from the downloaded repo, so the preprocessing (size, mean, std)
and the label names come from the repo's own config files.
"""
import glob
import hmac
import io
import json
import os
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from huggingface_hub import snapshot_download
from PIL import Image, UnidentifiedImageError

MODEL_REPO = os.getenv("MODEL_REPO", "JONNYVERSE/ai-image-detect-distilled-ONNX")
MODEL_NAME = os.getenv("MODEL_NAME", "vit-distilled-onnx")
MODEL_VERSION = os.getenv("MODEL_VERSION", "3")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
API_KEY = os.getenv("API_KEY", "")
MAX_BYTES = 5 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
AI_WORDS = ("ai", "fake", "artificial", "synthetic", "generated")

repoFolder = snapshot_download(MODEL_REPO)


def readJson(name):
    for path in glob.glob(os.path.join(repoFolder, "**", name), recursive=True):
        with open(path, encoding="utf-8") as openFile:
            return json.load(openFile)
    return {}


def pickModelFile():
    chosen = os.getenv("ONNX_FILE", "")
    if chosen:
        return os.path.join(repoFolder, chosen)
    files = sorted(glob.glob(os.path.join(repoFolder, "**", "*.onnx"), recursive=True))
    if not files:
        raise RuntimeError("No .onnx file found in " + MODEL_REPO)
    for path in files:
        lowered = os.path.basename(path).lower()
        if "int8" in lowered or "quant" in lowered:
            return path
    return files[0]


def pickAiIndex(labels):
    if os.getenv("AI_INDEX", "") != "":
        return int(os.environ["AI_INDEX"])
    for index, name in labels.items():
        if any(word in name.lower() for word in AI_WORDS):
            return index
    raise RuntimeError("Cannot tell which output means AI-generated. Labels: " + str(labels)
                       + ". Set the AI_INDEX environment variable.")


preConfig = readJson("preprocessor_config.json")
modelConfig = readJson("config.json")
labels = {int(key): value for key, value in modelConfig.get("id2label", {}).items()}
modelFile = pickModelFile()

sessionOptions = ort.SessionOptions()
sessionOptions.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
session = ort.InferenceSession(modelFile, sessionOptions, providers=["CPUExecutionProvider"])
inputName = session.get_inputs()[0].name
AI_INDEX = pickAiIndex(labels)

sizeValue = preConfig.get("size", 224)
if isinstance(sizeValue, dict):
    sizeValue = sizeValue.get("height") or sizeValue.get("shortest_edge") or 224
IMAGE_SIZE = int(sizeValue)
MEAN = np.array(preConfig.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(1, 1, 3)
STD = np.array(preConfig.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(1, 1, 3)
RESAMPLE = int(preConfig.get("resample", 2))      # 2 = bilinear in Pillow

app = FastAPI(title="VisionGuard ONNX model API")


def prepare(image):
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), RESAMPLE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.transpose(array, (2, 0, 1))[None].astype(np.float32)


def softmax(values):
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / exps.sum()


@app.get("/health")
def health():
    return {"model": MODEL_NAME, "status": "ok", "threshold": THRESHOLD, "version": MODEL_VERSION,
            "labels": labels, "ai_index": AI_INDEX, "file": os.path.basename(modelFile)}


@app.post("/predict")
def predict(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    if not API_KEY or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Missing or wrong API key")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 5 MB")
    try:
        image = Image.open(io.BytesIO(data))
        imageFormat = image.format
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The file cannot be decoded as an image") from None
    if imageFormat not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415, detail="Unsupported image type (allowed: JPEG, PNG, WEBP)")

    width, height = image.size
    started = time.perf_counter()
    logits = session.run(None, {inputName: prepare(image)})[0][0]
    probAi = float(softmax(np.asarray(logits, dtype=np.float64))[AI_INDEX])
    isAi = probAi >= THRESHOLD
    return {
        "label": "AI-generated" if isAi else "Real",
        "confidence": probAi if isAi else 1 - probAi,
        "prob_ai": probAi,
        "threshold": THRESHOLD,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "width": width,
        "height": height,
        "file_size": len(data),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
