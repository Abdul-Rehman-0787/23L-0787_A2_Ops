import sys
import numpy as np
import mlflow
from mlflow import MlflowClient
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

name = "visionguard"
client = MlflowClient()
mv = client.get_model_version_by_alias(name, "champion")
thr = client.get_run(mv.run_id).data.metrics["threshold"]
print(f"Loaded {name} version {mv.version} via alias 'champion', threshold {thr:.2f}")

model = mlflow.keras.load_model(f"models:/{name}@champion")
h, w = model.input_shape[1:3]
img = Image.open(sys.argv[1]).convert("RGB").resize((w, h))   # same resize as training
p = float(model.predict(np.array(img, dtype="float32")[None], verbose=0)[0][0])
print(f"P(AI-generated) = {p:.3f} -> {'AI-generated' if p >= thr else 'Real'}")