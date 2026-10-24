import matplotlib.pyplot as plt
from common import make_ds

x, y = next(iter(make_ds("train", training=True)))
plt.figure(figsize=(12, 8))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x[i].numpy().clip(0, 255).astype("uint8"))
    plt.title("AI" if y[i] == 1 else "Real"); plt.axis("off")
plt.tight_layout(); plt.savefig("reports/figures/augmented_batch.png", dpi=150)