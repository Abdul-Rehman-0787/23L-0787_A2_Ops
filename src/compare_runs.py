import mlflow, pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

runs = mlflow.search_runs(experiment_names=["visionguard"])
want = ["tags.mlflow.runName", "params.backbone", "params.lr_initial", "params.augmentation",
        "params.img_size", "params.unfreeze", "metrics.best_epoch", "metrics.val_loss_best",
        "metrics.val_roc_auc", "metrics.val_f1", "metrics.test_roc_auc", "metrics.test_f1"]
t = runs[[c for c in want if c in runs.columns]].copy()
t.columns = [c.split(".", 1)[1] for c in t.columns]
t = t.sort_values("val_roc_auc", ascending=False)
t.to_csv("reports/run_comparison.csv", index=False)
print(t.round(4).to_string(index=False))

# Use the actual column name produced by the rename step
ax = t.set_index("mlflow.runName")[["val_roc_auc", "val_f1"]].plot.barh(figsize=(9, 5))
ax.set_title("Run comparison (validation)"); plt.tight_layout()
plt.savefig("reports/figures/run_comparison.png", dpi=150)