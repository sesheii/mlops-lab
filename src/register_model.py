# src/register.py
import os
import mlflow
from mlflow.tracking import MlflowClient


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    model_name = "IMDB_RF_Model"

    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        print(f"Модель {model_name} не знайдена в Registry!")
        exit(1)

    latest_version = str(max(int(v.version) for v in versions))
    print(f"Знайдено останню версію: {latest_version}. Переводимо в Staging...")

    client.transition_model_version_stage(
        name=model_name, version=latest_version, stage="Staging"
    )


if __name__ == "__main__":
    main()
