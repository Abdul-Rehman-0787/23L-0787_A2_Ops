import io
import json
import os
import time

import numpy as np
import pandas as pd
import psutil
import redis
import tensorflow as tf
from PIL import Image

from api.preprocess import resizeBilinear
from app.auth import OtpService, hashPassword
from app.config import Config
from app.inference import Predictor
from app.knowledge import KnowledgeBase

process = psutil.Process(os.getpid())


def memoryMb():
    return round(process.memory_info().rss / 1e6)


def middle(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def high(values):
    ordered = sorted(values)
    position = int(len(ordered) * 0.95)
    if position > len(ordered) - 1:
        position = len(ordered) - 1
    return ordered[position]


def show(name, values):
    print(name, "| median", round(middle(values), 1), "ms | p95", round(high(values), 1),
          "ms | runs", len(values))


def millisecondsSince(start):
    return (time.perf_counter() - start) * 1000


print("Machine:", psutil.cpu_count(), "logical CPUs,",
      round(psutil.virtual_memory().total / 1e9), "GB RAM, CPU only")
print("Memory at start:", memoryMb(), "MB")

# ---- sample images (the first 30 test images that the app would accept)
samples = []
for path in pd.read_csv("data/splits/test.csv")["path"].tolist():
    with Image.open(path) as image:
        if image.size[0] * image.size[1] > 20_000_000:
            continue
    with open(path, "rb") as openFile:
        samples.append(openFile.read())
    if len(samples) == 30:
        break

# ---- 1. image model in the app (Keras, as loaded by INFERENCE_MODE=local)
predictor = Predictor(Config.MODEL_PATH, Config.THRESHOLD)
print("Memory after loading the Keras model (includes TensorFlow):", memoryMb(), "MB")
predictor.predict(samples[0])                       # warm-up
times = []
for data in samples:
    start = time.perf_counter()
    predictor.predict(data)
    times.append(millisecondsSince(start))
show("Inference, Keras in the app (decode, resize, predict)", times)

# ---- 2. image model in the cloud API (TFLite, one thread)
interpreter = tf.lite.Interpreter(model_path="api/model.tflite", num_threads=1)
interpreter.allocate_tensors()
inputInfo = interpreter.get_input_details()[0]
outputInfo = interpreter.get_output_details()[0]
height = int(inputInfo["shape"][1])
width = int(inputInfo["shape"][2])


def runTflite(data):
    with Image.open(io.BytesIO(data)) as image:
        pixels = resizeBilinear(np.asarray(image.convert("RGB")), height, width)
    interpreter.set_tensor(inputInfo["index"], pixels[None].astype("float32"))
    interpreter.invoke()
    return float(interpreter.get_tensor(outputInfo["index"])[0, 0])


runTflite(samples[0])
times = []
for data in samples:
    start = time.perf_counter()
    runTflite(data)
    times.append(millisecondsSince(start))
show("Inference, TFLite API path (decode, resize, predict)", times)

# ---- 3. one-time code and password hashing
try:
    client = redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)
    client.ping()
except redis.exceptions.RedisError:
    client = None
    print("OTP: Redis is not running, skipped")
if client is not None:
    otp = OtpService(client, "benchmark-secret-benchmark-secret-1234")
    times = []
    for number in range(20):
        email = "bench" + str(number) + "@example.com"
        start = time.perf_counter()
        otp.issueCode(email)
        times.append(millisecondsSince(start))
        for prefix in ["otp:", "otp_cool:", "otp_tries:", "otp_lock:"]:
            client.delete(prefix + email)
    show("One-time code (generate, hash, store in Redis; email delivery not included)", times)
times = []
for _ in range(5):
    start = time.perf_counter()
    hashPassword("benchmark password")
    times.append(millisecondsSince(start))
show("Password hashing (bcrypt)", times)

# ---- 4. document search
base = KnowledgeBase(Config.CHROMA_PATH, Config.EMBEDDING_CACHE, Config.KNOWLEDGE_FOLDER)
with open("tests/golden_questions.json", encoding="utf-8") as openFile:
    questions = json.load(openFile)
start = time.perf_counter()
base.search(questions[0]["question"], 3)
print("Document search, first call (loads the embedding model):",
      round(millisecondsSince(start)), "ms")
times = []
for item in questions:
    start = time.perf_counter()
    base.search(item["question"], 3)
    times.append(millisecondsSince(start))
show("Document search, warm", times)

print("Memory at the end (Keras + TFLite + embedding model + ChromaDB in one process):",
      memoryMb(), "MB")
print("Note: the web app in INFERENCE_MODE=remote does not load TensorFlow.")