import json
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

model = keras.models.load_model("models/transfer.keras")
thr = json.load(open("reports/transfer_threshold.json"))["threshold"]
rescale, base, gap, drop, dense = model.layers[1:6]

df = pd.read_csv("data/splits/test.csv")


def load(p):
    return np.array(Image.open(p).convert("RGB").resize((224, 224)), dtype="float32")


imgs = np.stack([load(p) for p in df["path"]])
probs = model.predict(imgs, batch_size=32).ravel()
df["prob"] = probs
df["correct"] = ((probs >= thr).astype(int) == df["label"])


def cam(img):
    x = tf.convert_to_tensor(img[None])
    with tf.GradientTape() as tape:
        conv = base(rescale(x), training=False)
        tape.watch(conv)
        pred = dense(gap(conv))
    g = tape.gradient(pred, conv)
    w = tf.reduce_mean(g, axis=(0, 1, 2))
    c = tf.nn.relu(tf.reduce_sum(conv[0] * w, axis=-1)).numpy()
    c = c / (c.max() + 1e-8)
    return np.array(Image.fromarray((c * 255).astype("uint8")).resize((224, 224)))/255.0


good = df[df["correct"]].sample(4, random_state=1)
bad = df[~df["correct"]].head(4)
fig, axes = plt.subplots(2, 8, figsize=(24, 7))
for j, (title, part) in enumerate([("CORRECT", good), ("WRONG", bad)]):
    for k, (i, r) in enumerate(part.iterrows()):
        col = j * 4 + k
        axes[0][col].imshow(imgs[i].astype("uint8")); axes[0][col].axis("off")
        axes[0][col].set_title(f"{title}\ntrue={'AI' if r.label else 'Real'} p={r.prob:.2f}", fontsize=9)
        axes[1][col].imshow(imgs[i].astype("uint8")); 
        axes[1][col].imshow(cam(imgs[i]), cmap="jet", alpha=0.45); axes[1][col].axis("off")
plt.tight_layout(); plt.savefig("reports/figures/gradcam.png", dpi=150)
print("wrong predictions available:", len(df[~df['correct']]))