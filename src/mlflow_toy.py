import argparse
import keras, mlflow
import matplotlib.pyplot as plt
from dotenv import load_dotenv

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, required=True)
p.add_argument("--dagshub", action="store_true")   # 3.2: log to DagsHub instead
a = p.parse_args()

if a.dagshub:
    load_dotenv()      # reads MLFLOW_* values from .env (never hard-code them)
else:
    # Use the exact absolute path to avoid any path resolution issues
    mlflow.set_tracking_uri("sqlite:///C:/Users/abdul/VS-CODE-FOLDER/23L-0787_A2_Ops/mlflow_toy.db")
    
mlflow.set_experiment("toy-mnist")

keras.utils.set_random_seed(42)
(x, y), _ = keras.datasets.mnist.load_data()
x, y = x[:20000] / 255.0, y[:20000]


class Log(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)


with mlflow.start_run(run_name=f"lr_{a.lr}"):
    mlflow.set_tag("purpose", "toy")
    mlflow.log_params({"lr": a.lr, "epochs": 5, "units": 64})
    model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                              keras.layers.Dense(64, activation="relu"),
                              keras.layers.Dense(10, activation="softmax")])
    model.compile(keras.optimizers.Adam(a.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(x, y, epochs=5, validation_split=0.2, callbacks=[Log()], verbose=2)

    fig = plt.figure()
    plt.plot(h.history["loss"], label="train"); plt.plot(h.history["val_loss"], label="val")
    plt.legend(); plt.title(f"Loss, lr={a.lr}")
    mlflow.log_figure(fig, "loss_curve.png")
    try:
        mlflow.keras.log_model(model, "model")
    except Exception as e:
        print("log_model failed:", e)