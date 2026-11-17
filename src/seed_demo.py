import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sqlalchemy import func, select

from api.preprocess import resizeBilinear
from app.config import Config
from app.db import Database, Prediction

maxPixels = 20_000_000

if len(sys.argv) < 2:
    print("Usage: python -m src.seed_demo demo@example.com   (the account must already exist)")
    sys.exit(1)
for needed in ["api/model.tflite", "api/model_quant.tflite", "models/champion.json",
               "data/splits/test.csv"]:
    if not os.path.exists(needed):
        print("Missing file:", needed, "- run python -m src.quantize_benchmark to create the quantized model")
        sys.exit(1)

email = sys.argv[1].strip().lower()
db = Database(Config.DATABASE_URL)
user = db.findUserByEmail(email)
if user is None:
    print("No account with that email. Register it in the app first.")
    sys.exit(1)

with db.Session() as session:
    already = session.scalar(select(func.count(Prediction.id)).where(
        Prediction.user_id == user.id, Prediction.model_version == "2-quant"))
if already:
    print("This account already has demo data. Nothing to do.")
    sys.exit(0)

with open("models/champion.json", encoding="utf-8") as openFile:
    threshold = float(json.load(openFile)["threshold"])


def makeInterpreter(path):
    interpreter = tf.lite.Interpreter(model_path=path, num_threads=2)
    interpreter.allocate_tensors()
    return interpreter


def runModel(interpreter, pixels):
    inputInfo = interpreter.get_input_details()[0]
    outputInfo = interpreter.get_output_details()[0]
    interpreter.set_tensor(inputInfo["index"], pixels[None].astype("float32"))
    interpreter.invoke()
    return float(interpreter.get_tensor(outputInfo["index"])[0, 0])


interpreters = {"2": makeInterpreter("api/model.tflite"),
                "2-quant": makeInterpreter("api/model_quant.tflite")}
inputShape = interpreters["2"].get_input_details()[0]["shape"]
height = int(inputShape[1])
width = int(inputShape[2])

saved = 0
skipped = 0
for path in pd.read_csv("data/splits/test.csv")["path"].tolist():
    with open(path, "rb") as openFile:
        rawBytes = openFile.read()
    with Image.open(path) as image:
        imageWidth, imageHeight = image.size
        if imageWidth * imageHeight > maxPixels:
            skipped = skipped + 1
            continue
        pixels = resizeBilinear(np.asarray(image.convert("RGB")), height, width)
    imageHash = hashlib.sha256(rawBytes).hexdigest()
    cleanName = "demo-"
    for character in os.path.basename(path):
        if character.isalnum() or character in "._-":
            cleanName = cleanName + character
        else:
            cleanName = cleanName + "_"
    for version in ["2", "2-quant"]:
        startTime = time.perf_counter()
        probAi = runModel(interpreters[version], pixels)
        milliseconds = (time.perf_counter() - startTime) * 1000
        isAi = probAi >= threshold
        db.add_prediction(
            user_id=user.id, filename=cleanName[:255], image_sha256=imageHash,
            label="AI-generated" if isAi else "Real",
            confidence=probAi if isAi else 1 - probAi, prob_ai=probAi, threshold=threshold,
            model_name="visionguard", model_version=version, width=imageWidth,
            height=imageHeight, file_size=len(rawBytes), latency_ms=milliseconds)
        saved = saved + 1

print("saved", saved, "demo analyses (skipped", skipped, "very large images)")
print("Now run: python -m src.graph_backfill")