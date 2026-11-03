import numpy as np
import tensorflow as tf
from api.preprocess import resizeBilinear


def testResizeMatchesTensorFlow():
    generator = np.random.default_rng(42)
    shapes = [(900, 1200, 3), (100, 80, 3), (224, 224, 3), (4000, 3000, 3)]
    for shape in shapes:
        image = generator.integers(0, 256, size=shape, dtype=np.uint8)
        mine = resizeBilinear(image, 224, 224)
        theirs = tf.image.resize(image, (224, 224)).numpy()
        difference = np.abs(mine - theirs)
        assert difference.max() < 0.1
        assert difference.mean() < 0.01