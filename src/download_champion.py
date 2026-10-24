import json, shutil
from pathlib import Path
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv
load_dotenv()

NAME = "visionguard"
client = MlflowClient()
mv = client.get_model_version_by_alias(NAME, "champion")
run = client.get_run(mv.run_id)

tmp = Path("models/_tmp")
src = mlflow.artifacts.download_artifacts(run_id=mv.run_id,
                                          artifact_path="model_file/model.keras",
                                          dst_path=str(tmp))
Path("models").mkdir(exist_ok=True)
shutil.copy(src, "models/champion.keras")
shutil.rmtree(tmp, ignore_errors=True)

thr = run.data.metrics["threshold"]
json.dump({"model_name": NAME, "version": mv.version, "run_id": mv.run_id,
           "threshold": thr}, open("models/champion.json", "w"), indent=2)
print("Saved models/champion.keras")
print(f"Put these in .env ->  MODEL_VERSION={mv.version}   THRESHOLD={thr:.2f}")