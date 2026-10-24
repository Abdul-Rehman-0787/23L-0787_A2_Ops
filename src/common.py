import tensorflow as tf
import pandas as pd
import keras

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.08),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
], name="augment")


def load_split(name):
    return pd.read_csv(f"data/splits/{name}.csv")


def make_ds(split, training=False, img_size=IMG_SIZE, use_aug=True):
    df = load_split(split)

    def _decode(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, img_size)
        return img, tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE).batch(BATCH)
    if training and use_aug:
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)