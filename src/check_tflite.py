import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from PIL import Image

m = keras.models.load_model("models/champion.keras")
it = tf.lite.Interpreter(model_path="api/model.tflite")
it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]
H, W = int(i["shape"][1]), int(i["shape"][2])

diffs = []
for p in pd.read_csv("data/splits/test.csv")["path"].head(10):
    raw = open(p, "rb").read()
    img = tf.image.resize(tf.image.decode_image(raw, channels=3, expand_animations=False), (H, W))
    k = float(m(img[None], training=False).numpy()[0, 0])
    x = np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.float32)[None]
    it.set_tensor(i["index"], x)
    it.invoke()
    t = float(it.get_tensor(o["index"])[0, 0])
    diffs.append(abs(k - t))
    print(f"keras={k:.4f}  api-style={t:.4f}  diff={abs(k - t):.4f}")
print("max diff:", round(max(diffs), 4))