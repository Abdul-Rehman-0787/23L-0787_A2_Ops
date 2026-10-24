import pandas as pd, matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("data/audit.csv")
df = df[~df.corrupt]
classes = sorted(df["class"].unique())

# Figure 1: sample grid, 8 per class
fig, axes = plt.subplots(len(classes), 8, figsize=(20, 5))
for r, c in enumerate(classes):
    sample = df[df["class"] == c].sample(8, random_state=42)
    for ax, p in zip(axes[r], sample["path"]):
        ax.imshow(Image.open(p).convert("RGB")); ax.axis("off")
    axes[r][0].set_title(c, loc="left")
plt.tight_layout(); plt.savefig("reports/figures/samples.png", dpi=150)

# Figures 2-4: histograms per class
for col in ["width", "height", "file_size"]:
    plt.figure(figsize=(7, 4))
    for c in classes:
        plt.hist(df[df["class"] == c][col], bins=40, alpha=0.5, label=c)
    plt.xlabel(col); plt.ylabel("count"); plt.legend()
    plt.title(f"{col} distribution per class")
    plt.savefig(f"reports/figures/hist_{col}.png", dpi=150)
print("done")



