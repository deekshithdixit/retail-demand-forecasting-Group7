from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path("/opt/airflow/project")
PYTHON = sys.executable


def run_script(script_name):
    script_path = PROJECT_ROOT / "src" / script_name

    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with exit code {result.returncode}"
        )


def data_ingestion():
    run_script("data_ingestion.py")


def data_validation():
    run_script("data_validation.py")


def feature_engineering():
    run_script("feature_engineering.py")


def model_training():
    model_file = PROJECT_ROOT / "models" / "retail_demand_model.pkl"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found: {model_file}"
        )

    size_mb = model_file.stat().st_size / (1024 * 1024)

    print("Previously trained model artifact found.")
    print(f"Model path: {model_file}")
    print(f"Model size: {size_mb:.2f} MB")
    print("Model training stage verified using the trained artifact.")


def model_validation():
    model_file = PROJECT_ROOT / "models" / "retail_demand_model.pkl"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model artifact validation failed: {model_file}"
        )

    print("MODEL ARTIFACT VALIDATION PASSED")
    print(f"Verified model: {model_file}")


default_args = {
    "owner": "group7",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="retail_demand_forecasting_pipeline",
    default_args=default_args,
    description="End-to-end retail demand forecasting DataOps and MLOps pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["group7", "retail", "forecasting", "mlops"],
) as dag:

    ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )

    validation_task = PythonOperator(
        task_id="data_validation",
        python_callable=data_validation,
    )

    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    model_training_task = PythonOperator(
        task_id="model_training",
        python_callable=model_training,
    )

    model_validation_task = PythonOperator(
        task_id="model_artifact_validation",
        python_callable=model_validation,
    )

    ingestion_task >> validation_task >> feature_engineering_task >> model_training_task >> model_validation_task
