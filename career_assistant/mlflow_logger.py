import mlflow
from mlflow import log_metric, log_param, log_artifact
import os
from contextlib import contextmanager

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("career_assistant_experiment")


@contextmanager
def start_run(run_name=None, tags=None):
    """
    Context manager to automatically start and end MLflow run.
    
    Usage:
        with start_run("baseline_model") as run:
            log_param("model", "logistic_regression")
            log_metric("accuracy", 0.85)
    """
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def log_params(params: dict):
    """Log multiple parameters at once."""
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics: dict):
    """Log multiple metrics at once."""
    for k, v in metrics.items():
        mlflow.log_metric(k, v)


def log_artifacts_from_path(path: str):
    """Log files/folders as artifacts."""
    if os.path.exists(path):
        mlflow.log_artifact(path)
    else:
        raise FileNotFoundError(f"Path {path} does not exist.")


def log_model(model, artifact_path="models", registered_model_name=None):
    """
    Log a model to MLflow.
    Supports sklearn, pytorch, or generic models with mlflow.sklearn/torch/pickle.
    """
    try:
        import sklearn
        mlflow.sklearn.log_model(model, artifact_path=artifact_path, registered_model_name=registered_model_name)
    except Exception:
        try:
            import torch
            mlflow.pytorch.log_model(model, artifact_path=artifact_path, registered_model_name=registered_model_name)
        except Exception:
            import pickle
            model_path = os.path.join(artifact_path, "model.pkl")
            os.makedirs(artifact_path, exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, artifact_path=artifact_path)
