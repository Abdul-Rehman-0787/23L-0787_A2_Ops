import argparse, re, subprocess, sys
from pathlib import Path
import keras, mlflow, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from keras import layers
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay)
from common import make_ds, load_split, SEED

load_dotenv()
p = argparse.ArgumentParser()
p.add_argument("--run-name", required=True)
p.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0"])
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--img-size", type=int, default=224)
p.add_argument("--no-aug", action="store_true")
p.add_argument("--unfreeze", type=int, default=0)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--logging", choices=["manual", "autolog"], default="manual")
p.add_argument("--experiment", default="visionguard")
a = p.parse_args()

keras.utils.set_random_seed(SEED)
size, shape = (a.img_size, a.img_size), (a.img_size, a.img_size, 3)
out = Path("reports/runs") / a.run_name
out.mkdir(parents=True, exist_ok=True)


def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"


def dvc_md5():
    m = re.search(r"md5:\s*(\S+)", Path("data/raw.dvc").read_text())
    return m.group(1) if m else "unknown"


def build():
    if a.backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)
        pre = layers.Rescaling(1 / 127.5, offset=-1.0)     # MobileNetV2 expects [-1, 1]
    else:
        base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=shape)
        pre = layers.Identity()                            # EfficientNet rescales internally (0-255 in)
    base.trainable = a.unfreeze > 0
    if a.unfreeze > 0:
        for l in base.layers[:-a.unfreeze]:
            l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    inp = keras.Input(shape)
    x = base(pre(inp), training=False)
    x = layers.Dropout(a.dropout)(layers.GlobalAveragePooling2D()(x))
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x))


class MlflowLog(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)


def scores(y, probs, thr):
    pred = (probs >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, probs)}


if a.logging == "autolog":
    try:
        mlflow.tensorflow.autolog(log_models=False)
    except Exception as e:
        print("autolog unavailable:", e)
mlflow.set_experiment(a.experiment)

train_ds = make_ds("train", True, size, not a.no_aug)
val_ds, test_ds = make_ds("val", False, size), make_ds("test", False, size)

with mlflow.start_run(run_name=a.run_name) as run:
    mlflow.set_tags({"git_commit": sh(["git", "rev-parse", "--short", "HEAD"]),
                     "dvc_data_md5": dvc_md5(), "seed": str(SEED),
                     "logging_mode": a.logging, "framework": "keras"})
    mlflow.log_params({"backbone": a.backbone, "lr_initial": a.lr, "img_size": a.img_size,
                       "augmentation": not a.no_aug, "unfreeze": a.unfreeze,
                       "dropout": a.dropout, "max_epochs": a.epochs})
    model = build()
    model.compile(keras.optimizers.Adam(a.lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    cbs = [keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-7)]
    if a.logging == "manual":
        cbs.append(MlflowLog())
    hist = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cbs, verbose=2).history
    mlflow.log_metric("best_epoch", int(np.argmin(hist["val_loss"])) + 1)
    mlflow.log_metric("val_loss_best", float(min(hist["val_loss"])))

    # threshold chosen on VALIDATION; test used only for reporting
    yv, pv = load_split("val")["label"].values, model.predict(val_ds, verbose=0).ravel()
    grid = np.linspace(0.05, 0.95, 91)
    thr = float(grid[int(np.argmax([f1_score(yv, (pv >= t).astype(int)) for t in grid]))])
    mlflow.log_metric("threshold", thr)
    mlflow.log_metrics({f"val_{k}": v for k, v in scores(yv, pv, thr).items()})

    yt, pt = load_split("test")["label"].values, model.predict(test_ds, verbose=0).ravel()
    mlflow.log_metrics({f"test_{k}": v for k, v in scores(yt, pt, thr).items()})

    cm = confusion_matrix(yt, (pt >= thr).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["Real", "AI"]).plot()
    plt.title(a.run_name); plt.savefig(out / "confusion_matrix.png", dpi=120); plt.close()
    fpr, tpr, _ = roc_curve(yt, pt)
    plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--"); plt.title(f"ROC {a.run_name}")
    plt.savefig(out / "roc_curve.png", dpi=120); plt.close()
    (out / "requirements.txt").write_text(sh([sys.executable, "-m", "pip", "freeze"]))
    model.save(out / "model.keras")

    mlflow.log_artifact(str(out / "confusion_matrix.png"), "plots")
    mlflow.log_artifact(str(out / "roc_curve.png"), "plots")
    mlflow.log_artifact(str(out / "requirements.txt"), "environment")
    mlflow.log_artifacts("data/splits", "splits")
    mlflow.log_artifact(str(out / "model.keras"), "model_file")
    try:
        mlflow.keras.log_model(model, "model")     # MLflow model format, needed for the registry
    except Exception as e:
        print("mlflow.keras.log_model failed:", e)
    (out / "run_id.txt").write_text(run.info.run_id)
    print("RUN ID:", run.info.run_id, "| threshold:", thr)