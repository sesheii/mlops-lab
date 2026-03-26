from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
from docker.types import Mount
import os
import json

HOST_PROJECT_PATH = os.getenv("HOST_PROJECT_PATH", "D:/repos/mlops_lab_1")
ACCURACY_THRESHOLD = 0.8

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 3, 24),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def check_model_quality():
    metrics_path = "/opt/airflow/reports/metrics.json"

    if not os.path.exists(metrics_path):
        print(f"File not found: {metrics_path}")
        return "model_rejected"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    accuracy = metrics.get("accuracy", 0)
    print(f"Model accuracy: {accuracy}")

    if accuracy >= ACCURACY_THRESHOLD:
        return "register_model"
    else:
        return "model_rejected"


with DAG(
    "ml_training_pipeline",
    default_args=default_args,
    description="MLOps Pipeline: Full DVC Flow",
    schedule=timedelta(days=1),
    catchup=False,
) as dag:

    venv_mount = Mount(source="", target="/app/.venv", type="volume")
    project_mount = Mount(source=HOST_PROJECT_PATH, target="/app", type="bind")

    docker_kwargs = {
        "image": "mlops-lab-model:latest",
        "docker_url": "unix://var/run/docker.sock",
        "network_mode": "mlops-network",
        "auto_remove": "force",
        "mounts": [project_mount, venv_mount],
    }

    check_dvc = DockerOperator(
        task_id="check_dvc_status",
        command="rm -f .dvc/tmp/rwlock && dvc status",
        **docker_kwargs,
    )

    download_data = DockerOperator(
        task_id="download_data", command="dvc repro download", **docker_kwargs
    )

    prepare_data = DockerOperator(
        task_id="prepare_data", command="dvc repro prepare", **docker_kwargs
    )

    test_data = DockerOperator(
        task_id="test_data", command="dvc repro test_data", **docker_kwargs
    )

    train_model = DockerOperator(
        task_id="train_model",
        command="dvc repro optimize",
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow-server:5000",
            "UV_LINK_MODE": "copy",
        },
        **docker_kwargs,
    )

    test_model = DockerOperator(
        task_id="test_model", command="dvc repro test_model", **docker_kwargs
    )

    branching = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=check_model_quality,
    )

    register_model = DockerOperator(
        task_id="register_model",
        command="mlflow models register -m 'models:/IMDB_RF_Model/latest' -n IMDB_Production_Model",
        environment={"MLFLOW_TRACKING_URI": "http://mlflow-server:5000"},
        **docker_kwargs,
    )

    model_rejected = EmptyOperator(task_id="model_rejected")

    (
        check_dvc
        >> download_data
        >> prepare_data
        >> test_data
        >> train_model
        >> test_model
        >> branching
    )
    branching >> [register_model, model_rejected]
