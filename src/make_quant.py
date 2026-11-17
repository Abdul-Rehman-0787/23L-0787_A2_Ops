import keras
import tensorflow as tf

model = keras.models.load_model("models/champion.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
with open("api/model_quant.tflite", "wb") as openFile:
    openFile.write(converter.convert())
print("saved api/model_quant.tflite")