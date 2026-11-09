import json

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from api.preprocess import resizeBilinear

with open("models/champion.json", encoding="utf-8") as fh:
    threshold = json.load(fh)["threshold"]
kerasModel = keras.models.load_model("models/champion.keras")
interpreter = tf.lite.Interpreter(model_path="api/model.tflite")
interpreter.allocate_tensors()
inputInfo = interpreter.get_input_details()[0]
outputInfo = interpreter.get_output_details()[0]
height = int(inputInfo["shape"][1])
width = int(inputInfo["shape"][2])


def runTflite(pixels):
    interpreter.set_tensor(inputInfo["index"], pixels[None].astype("float32"))
    interpreter.invoke()
    return float(interpreter.get_tensor(outputInfo["index"])[0, 0])


names = ["tfliteTf", "tflitePil", "tfliteNumpy"]
diffs = {name: [] for name in names}
flips = {name: 0 for name in names}
paths = pd.read_csv("data/splits/test.csv")["path"].tolist()

for path in paths:
    with open(path, "rb") as fh:
        rawBytes = fh.read()
    decoded = tf.image.decode_image(rawBytes, channels=3, expand_animations=False)
    tfPixels = tf.image.resize(decoded, (height, width)).numpy()
    kerasProb = float(kerasModel(tfPixels[None], training=False).numpy()[0, 0])

    pilImage = Image.open(path).convert("RGB")
    pilPixels = np.asarray(pilImage.resize((width, height), Image.BILINEAR), dtype="float32")
    numpyPixels = resizeBilinear(np.asarray(pilImage), height, width)

    probs = {"tfliteTf": runTflite(tfPixels),
             "tflitePil": runTflite(pilPixels),
             "tfliteNumpy": runTflite(numpyPixels)}
    for name in names:
        diffs[name].append(abs(probs[name] - kerasProb))
        if (probs[name] >= threshold) != (kerasProb >= threshold):
            flips[name] += 1

print("images:", len(paths), "| threshold:", threshold)
for name in names:
    print(name, "mean diff", round(float(np.mean(diffs[name])), 4),
          "max diff", round(float(np.max(diffs[name])), 4), "label flips", flips[name])