import json
import subprocess
import time

import keras
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import roc_auc_score

from api.preprocess import resizeBilinear

load_dotenv()
floatPath = "api/model.tflite"
quantPath = "api/model_quant.tflite"
championInfo = json.load(open("models/champion.json"))
threshold = float(championInfo["threshold"])

# 1. Build the quantized model (dynamic-range: weights stored as 8-bit)
kerasModel = keras.models.load_model("models/champion.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(kerasModel)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantBytes = converter.convert()
open(quantPath, "wb").write(quantBytes)


def makeInterpreter(path):
    interpreter = tf.lite.Interpreter(model_path=path, num_threads=1)
    interpreter.allocate_tensors()
    return interpreter


def runInterpreter(interpreter, pixels):
    inputInfo = interpreter.get_input_details()[0]
    outputInfo = interpreter.get_output_details()[0]
    interpreter.set_tensor(inputInfo["index"], pixels[None].astype("float32"))
    interpreter.invoke()
    return float(interpreter.get_tensor(outputInfo["index"])[0, 0])


floatInterpreter = makeInterpreter(floatPath)
quantInterpreter = makeInterpreter(quantPath)
inputShape = floatInterpreter.get_input_details()[0]["shape"]
height = int(inputShape[1])
width = int(inputShape[2])

# 2. Score all test images
testTable = pd.read_csv("data/splits/test.csv")
paths = testTable["path"].tolist()
labels = testTable["label"].tolist()
kerasProbs = []
floatProbs = []
quantProbs = []
firstPixels = None

for path in paths:
    rawBytes = open(path, "rb").read()
    decoded = tf.image.decode_image(rawBytes, channels=3, expand_animations=False)
    tfPixels = tf.image.resize(decoded, (height, width)).numpy()
    kerasProbs.append(float(kerasModel(tfPixels[None], training=False).numpy()[0, 0]))

    # same steps as the deployed API
    apiPixels = resizeBilinear(np.asarray(Image.open(path).convert("RGB")), height, width)
    floatProbs.append(runInterpreter(floatInterpreter, apiPixels))
    quantProbs.append(runInterpreter(quantInterpreter, apiPixels))
    if firstPixels is None:
        firstPixels = apiPixels


def scoreModel(probs):
    correct = 0
    flips = 0
    totalGap = 0.0
    biggestGap = 0.0
    for i in range(len(probs)):
        predictedAi = probs[i] >= threshold
        if predictedAi == (labels[i] == 1):
            correct += 1
        if predictedAi != (kerasProbs[i] >= threshold):
            flips += 1
        gap = abs(probs[i] - kerasProbs[i])
        totalGap += gap
        if gap > biggestGap:
            biggestGap = gap
    return {"accuracy": correct / len(probs), "auc": roc_auc_score(labels, probs),
            "flips": flips, "meanDiff": totalGap / len(probs), "maxDiff": biggestGap}


def measureLatency(interpreter, runs=50):
    for _ in range(5):
        runInterpreter(interpreter, firstPixels)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        runInterpreter(interpreter, firstPixels)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[len(times) // 2]


results = {"keras": scoreModel(kerasProbs), "float": scoreModel(floatProbs), "quant": scoreModel(quantProbs)}
sizes = {"float": len(open(floatPath, "rb").read()) / 1e6, "quant": len(quantBytes) / 1e6}
latency = {"float": measureLatency(floatInterpreter), "quant": measureLatency(quantInterpreter)}

print("threshold", threshold, "| test images", len(paths))
for name in ["keras", "float", "quant"]:
    r = results[name]
    print(name, "acc", round(r["accuracy"], 3), "auc", round(r["auc"], 3),
          "flips", r["flips"], "meanDiff", round(r["meanDiff"], 4), "maxDiff", round(r["maxDiff"], 4))
for name in ["float", "quant"]:
    print(name, "size MB", round(sizes[name], 2), "| median latency ms (1 thread)", round(latency[name], 1))

# 3. Log to a new experiment (assignment runs stay untouched)
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
mlflow.set_experiment("visionguard-platform")
with mlflow.start_run(run_name="tflite-quantization"):
    mlflow.log_param("quantization", "dynamic-range")
    mlflow.log_param("champion_version", championInfo.get("version", "unknown"))
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("resize", "numpy bilinear, TensorFlow-exact")
    mlflow.set_tag("git_commit", commit)
    for name in ["keras", "float", "quant"]:
        for key, value in results[name].items():
            mlflow.log_metric(name + "_" + key, float(value))
    for name in ["float", "quant"]:
        mlflow.log_metric(name + "_size_mb", sizes[name])
        mlflow.log_metric(name + "_latency_ms", latency[name])
    mlflow.log_artifact(quantPath)
print("logged to MLflow experiment visionguard-platform")