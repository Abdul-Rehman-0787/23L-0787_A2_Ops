import argparse
import json

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from common import load_split, make_ds

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)          # baseline or transfer
p.add_argument("--test", action="store_true")     # run ONLY once, at the end
p.add_argument("--beta", type=float, default=1.0)
args = p.parse_args()

model = keras.models.load_model(f"models/{args.model}.keras")
split = "test" if args.test else "val"
df = load_split(split)
probs = model.predict(make_ds(split, False)).ravel()   # same order as the CSV
y = df["label"].values


def metrics(thr):
    pred = (probs >= thr).astype(int)
    return {"threshold": float(thr),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, probs)}


if not args.test:
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y, (probs >= t).astype(int), beta=args.beta) for t in grid]
    thr = float(grid[int(np.argmax(scores))])
    with open(f"reports/{args.model}_threshold.json", "w", encoding="utf-8") as fh:
        json.dump({"threshold": thr, "beta": args.beta}, fh)
    print(f"Chosen threshold on VALIDATION (beta={args.beta}):", thr)
    print(json.dumps(metrics(thr), indent=2))
else:
    with open(f"reports/{args.model}_threshold.json", encoding="utf-8") as fh:
        thr = json.load(fh)["threshold"]
    m = metrics(thr)
    print(json.dumps(m, indent=2))
    print("For reference, threshold 0.5:", json.dumps(metrics(0.5), indent=2))
    with open(f"reports/{args.model}_test_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(m, fh)

    pred = (probs >= thr).astype(int)
    cm = confusion_matrix(y, pred)
    print("Confusion matrix [[TN FP][FN TP]]:\n", cm)
    ConfusionMatrixDisplay(cm, display_labels=["Real", "AI"]).plot()
    plt.title(f"{args.model} confusion matrix (test, thr={thr:.2f})")
    plt.savefig(f"reports/figures/{args.model}_confusion.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(y, probs)
    plt.plot(fpr, tpr, label=f"AUC = {m['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"{args.model} ROC (test)"); plt.legend()
    plt.savefig(f"reports/figures/{args.model}_roc.png", dpi=150); plt.close()

    out = df.assign(prob=probs, pred=pred)
    out[out["pred"] != out["label"]].to_csv(f"reports/{args.model}_errors.csv", index=False)