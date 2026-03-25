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
    metrics_path = os.path.join(HOST_PROJECT_PATH, "reports", "metrics.json")

    if not os.path.exists(metrics_path):
        return "model_rejected"

    with open(metrics_path, "r") as f:
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
    description="MLOps Pipeline: Check, Prepare, Train",
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    venv_mount = Mount(source="", target="/app/.venv", type="volume")

    project_mount = Mount(source=HOST_PROJECT_PATH, target="/app", type="bind")

    check_data = DockerOperator(
        task_id="check_dvc_status",
        image="mlops-lab-model:latest",
        # command='dvc status',
        command="rm -f .dvc/tmp/rwlock && dvc status",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove="force",
        mounts=[project_mount, venv_mount],
    )

    prepare_data = DockerOperator(
        task_id="prepare_data",
        image="mlops-lab-model:latest",
        command="dvc repro prepare",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove="force",
        mounts=[project_mount, venv_mount],
    )

    train_model = DockerOperator(
        task_id="train_model",
        image="mlops-lab-model:latest",
        command="dvc repro optimize",
        docker_url="unix://var/run/docker.sock",
        network_mode="mlops-network",
        auto_remove="force",
        mounts=[project_mount, venv_mount],
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow-server:5000",
            "UV_LINK_MODE": "copy",
        },
    )

    branching = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=check_model_quality,
    )

    register_model = DockerOperator(
        task_id="register_model",
        image="mlops-lab-model:latest",
        command="mlflow models register -m 'models:/IMDB_RF_Model/latest' -n IMDB_Production_Model",
        docker_url="unix://var/run/docker.sock",
        network_mode="mlops-network",
        auto_remove="force",
        environment={"MLFLOW_TRACKING_URI": "http://mlflow-server:5000"},
    )

    model_rejected = EmptyOperator(task_id="model_rejected")

    check_data >> prepare_data >> train_model >> branching
    branching >> [register_model, model_rejected]
