import shutil
from pathlib import Path
import keras
import tensorflow as tf

Path("api").mkdir(exist_ok=True)
m = keras.models.load_model("models/champion.keras")
tfl = tf.lite.TFLiteConverter.from_keras_model(m).convert()
Path("api/model.tflite").write_bytes(tfl)
shutil.copy("models/champion.json", "api/champion.json")
print("input shape:", m.input_shape, "| tflite MB:", round(len(tfl) / 1e6, 1))