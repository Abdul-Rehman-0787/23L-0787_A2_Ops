import argparse, json, shutil
from pathlib import Path
import keras
import matplotlib.pyplot as plt
import numpy as np
from common import make_ds, SEED
from models import (build_baseline, build_transfer, unfreeze_last_block,
                    trainable_report)

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["baseline", "transfer"], required=True)
p.add_argument("--epochs", type=int, default=30)
args = p.parse_args()

keras.utils.set_random_seed(SEED)      # Python, NumPy and TF seeds
Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)
train_ds, val_ds = make_ds("train", True), make_ds("val", False)


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])


def fit(model, name, epochs):
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(f"models/{name}.keras",
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-7),
    ]
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)
    hist = h.history
    best = int(np.argmin(hist["val_loss"])) + 1
    print(f"[{name}] best epoch = {best}, val_loss = {min(hist['val_loss']):.4f}")
    json.dump({"history": hist, "best_epoch": best},
              open(f"reports/{name}_history.json", "w"))
    plot(hist, best, name)
    return min(hist["val_loss"])


def plot(hist, best, name):
    ep = range(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k, t in zip(ax, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        a.plot(ep, hist[k], label="train")
        a.plot(ep, hist["val_" + k], label="val")
        a.axvline(best, color="red", ls="--", label=f"best epoch {best}")
        a.set_xlabel("epoch"); a.set_title(f"{name}: {t}"); a.legend()
    plt.tight_layout(); plt.savefig(f"reports/figures/{name}_curves.png", dpi=150)


if args.model == "baseline":
    model = build_baseline()
    model.summary()
    compile_model(model, 1e-3)
    fit(model, "baseline", args.epochs)
else:
    model, base = build_transfer()
    model.summary()
    print("\n=== STAGE 1: frozen backbone ===")
    trainable_report(model, base)
    compile_model(model, 1e-3)
    loss1 = fit(model, "transfer_stage1", args.epochs)

    print("\n=== STAGE 2: unfreeze last block ===")
    unfreeze_last_block(base)
    trainable_report(model, base)
    compile_model(model, 1e-5)         # much smaller learning rate
    loss2 = fit(model, "transfer_stage2", args.epochs)

    winner = "transfer_stage2" if loss2 < loss1 else "transfer_stage1"
    shutil.copy(f"models/{winner}.keras", "models/transfer.keras")
    print("Final transfer model =", winner)