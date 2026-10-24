import sys
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

name = sys.argv[1] if len(sys.argv) > 1 else "transfer"
err = pd.read_csv(f"reports/{name}_errors.csv")
err["conf"] = (err["prob"] - err["label"]).abs()      # how wrong it was
err = err.sort_values("conf", ascending=False).head(10)

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for ax, (_, r) in zip(axes.ravel(), err.iterrows()):
    ax.imshow(Image.open(r["path"]).convert("RGB")); ax.axis("off")
    true = "AI" if r["label"] == 1 else "Real"
    ax.set_title(f"true={true}  p(AI)={r['prob']:.2f}\n{r['width']}x{r['height']}", fontsize=9)
plt.tight_layout(); plt.savefig(f"reports/figures/{name}_errors.png", dpi=150)
print(err[["path", "label", "prob", "width", "height", "file_size"]])