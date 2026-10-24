import keras
import keras_tuner as kt
import pandas as pd
from keras import layers
from common import make_ds, SEED

SHAPE = (224, 224, 3)
keras.utils.set_random_seed(SEED)
train_ds, val_ds = make_ds("train", True), make_ds("val", False)


def build(hp):
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                          input_shape=SHAPE)
    n_unfreeze = hp.Choice("unfrozen_layers", [0, 10, 30, 60])
    base.trainable = n_unfreeze > 0
    if n_unfreeze > 0:
        for l in base.layers[:-n_unfreeze]:
            l.trainable = False
        for l in base.layers:     # keep BatchNorm frozen
            if isinstance(l, layers.BatchNormalization):
                l.trainable = False

    inp = keras.Input(SHAPE)
    x = layers.Rescaling(1 / 127.5, offset=-1.0)(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(hp.Choice("dropout", [0.2, 0.3, 0.5]))(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(keras.optimizers.Adam(hp.Float("lr", 1e-5, 1e-3, sampling="log")),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


tuner = kt.RandomSearch(build, objective="val_loss", max_trials=10,
                        directory="tuner_results", project_name="visionguard",
                        overwrite=True, seed=SEED)
tuner.search(train_ds, validation_data=val_ds, epochs=6,
             callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=2)])

rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_loss": t.score}
        for t in tuner.oracle.get_best_trials(10)]
table = pd.DataFrame(rows)
table.to_csv("reports/tuner_results.csv", index=False)
print("\n", table.to_string(index=False))
print("\nBest settings:", tuner.get_best_hyperparameters(1)[0].values)

for col in ["unfrozen_layers", "dropout"]:
    print(f"\nMean val_loss by {col}:\n", table.groupby(col)["val_loss"].mean())
print("\nCorrelation of log(lr) with val_loss:",
      round(table["lr"].apply(lambda v: __import__("math").log10(v)).corr(table["val_loss"]), 3))