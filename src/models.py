import numpy as np
import keras
from keras import layers

SHAPE = (224, 224, 3)


def build_baseline():
    inp = keras.Input(SHAPE)
    x = layers.Rescaling(1 / 255.0)(inp)
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="baseline_cnn")


def build_transfer():
    base = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=SHAPE)
    base.trainable = False
    inp = keras.Input(SHAPE)
    # MobileNetV2 expects pixels in [-1, 1]: x / 127.5 - 1
    x = layers.Rescaling(1 / 127.5, offset=-1.0)(inp)
    x = base(x, training=False)          # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="mobilenetv2_transfer"), base


def unfreeze_last_block(base):
    base.trainable = True
    for l in base.layers:
        in_last_block = l.name.startswith(("block_16", "Conv_1"))
        l.trainable = in_last_block and not isinstance(l, layers.BatchNormalization)


def trainable_report(model, base=None):
    top = [l.name for l in model.layers if l.trainable and l.weights and l is not base]
    inner = [l.name for l in base.layers if l.trainable and l.weights] if base else []
    n = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print("Trainable head layers:", top)
    print("Trainable backbone layers:", inner)
    print("Trainable parameters:", n)