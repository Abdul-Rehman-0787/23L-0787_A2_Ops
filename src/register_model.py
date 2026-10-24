import sys
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

# Load environment variables (MLFLOW_TRACKING_URI, etc.)
load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python src/register_model.py <run_id> [alias]")
    sys.exit(1)

run_id = sys.argv[1]
alias = sys.argv[2] if len(sys.argv) > 2 else None
name = "visionguard"

client = MlflowClient()

# 1. Dynamically find the model logged in this run
# MLflow 3.x stores models as separate entities (LoggedModel)
try:
    # Get the run details to find the experiment_id
    run = client.get_run(run_id)
    exp_id = run.info.experiment_id
    
    # Search for models logged in this experiment
    logged_models = client.search_logged_models(experiment_ids=[exp_id])
    
    # Find the one linked to our specific run_id
    target_model = None
    for model in logged_models:
        if model.source_run_id == run_id:
            target_model = model
            break
            
    if not target_model:
        raise ValueError(f"No model found logged under run {run_id}")
        
    # 2. Get the dynamic model URI
    model_uri = target_model.model_uri
    print(f"Found model: {target_model.name}")
    print(f"Model URI: {model_uri}")

    # 3. Register the model using the dynamic URI
    mv = mlflow.register_model(model_uri=model_uri, name=name)
    print(f"Registered {name} version {mv.version}")

    # 4. Set alias if provided
    if alias:
        client.set_registered_model_alias(name, alias, mv.version)
        print(f"Alias '{alias}' -> version {mv.version}")

except Exception as e:
    print(f"Error during registration: {e}")