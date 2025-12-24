import os
import mlflow
import pandas as pd

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("career_assistant_data_pipeline")


def log_dataframe(df: pd.DataFrame, artifact_path: str, filename: str):
    """
    Save a DataFrame as CSV and log it as an MLflow artifact.
    """
    os.makedirs(artifact_path, exist_ok=True)
    file_path = os.path.join(artifact_path, filename)
    df.to_csv(file_path, index=False)
    mlflow.log_artifact(file_path, artifact_path=artifact_path)


def start_data_run(run_name="data_pipeline_run"):
    """
    Context manager to track preprocessing runs.
    Usage:
        with start_data_run("preprocess_v1") as run:
            log_dataframe(df, "processed_data", "df_v1.csv")
    """
    return mlflow.start_run(run_name=run_name)
