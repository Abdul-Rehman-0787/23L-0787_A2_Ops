import numpy as np


def resizeBilinear(image, outHeight, outWidth):
    """Resize like tf.image.resize: bilinear, no smoothing, half-pixel centres.
    image is an array shaped (height, width, 3). Returns float32."""
    inHeight = image.shape[0]
    inWidth = image.shape[1]
    ys = (np.arange(outHeight) + 0.5) * inHeight / outHeight - 0.5
    xs = (np.arange(outWidth) + 0.5) * inWidth / outWidth - 0.5

    top = np.clip(np.floor(ys), 0, inHeight - 1).astype(int)
    bottom = np.clip(np.ceil(ys), 0, inHeight - 1).astype(int)
    left = np.clip(np.floor(xs), 0, inWidth - 1).astype(int)
    right = np.clip(np.ceil(xs), 0, inWidth - 1).astype(int)

    weightY = (ys - np.floor(ys))[:, None, None]
    weightX = (xs - np.floor(xs))[None, :, None]

    topRows = image[top].astype("float32")
    bottomRows = image[bottom].astype("float32")
    topMix = topRows[:, left] * (1 - weightX) + topRows[:, right] * weightX
    bottomMix = bottomRows[:, left] * (1 - weightX) + bottomRows[:, right] * weightX
    return (topMix * (1 - weightY) + bottomMix * weightY).astype("float32")