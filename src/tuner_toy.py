import keras
import keras_tuner as kt
import pandas as pd

keras.utils.set_random_seed(42)
(x_tr, y_tr), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()
x_tr, x_te = x_tr / 255.0, x_te / 255.0


def build(hp):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model.compile(keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=6,
                        directory="tuner_results", project_name="toy_fashion",
                        overwrite=True, seed=42)
tuner.search(x_tr, y_tr, epochs=3, validation_split=0.2, verbose=1)

rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score}
        for t in tuner.oracle.get_best_trials(6)]
table = pd.DataFrame(rows)
print("\n", table.to_string(index=False))
best = tuner.get_best_hyperparameters(1)[0]
print("\nBest settings:", best.values)