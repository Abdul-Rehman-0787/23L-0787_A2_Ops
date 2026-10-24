import threading

import keras
import tensorflow as tf


class Predictor:
    def __init__(self, model_path, threshold):
        try:
            self.model = keras.models.load_model(model_path)   # loaded ONCE, at start-up
        except Exception as e:
            raise RuntimeError(f"Cannot load model from '{model_path}': {e}") from e
        self.threshold = threshold
        self.size = tuple(self.model.input_shape[1:3])
        self._lock = threading.Lock()

    def predict(self, data: bytes):
        # SAME ops as training: decode_image(channels=3) then tf.image.resize
        img = tf.image.decode_image(data, channels=3, expand_animations=False)
        img = tf.image.resize(img, self.size)                  # float32, 0-255
        with self._lock:
            p = float(self.model(img[None], training=False).numpy()[0, 0])
        is_ai = p >= self.threshold
        return {"label": "AI-generated" if is_ai else "Real",
                "confidence": p if is_ai else 1 - p,
                "prob_ai": p, "threshold": self.threshold}


if __name__ == "__main__":          # python -m app.inference img1 img2
    import sys

    from .config import Config

    pred = Predictor(Config.MODEL_PATH, Config.THRESHOLD)
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            r = pred.predict(f.read())
        print(f"{path} -> {r['label']} (confidence {r['confidence']:.1%}, P(AI)={r['prob_ai']:.3f})")