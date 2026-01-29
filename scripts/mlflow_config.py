#!/usr/bin/env python3
"""
MLflow Configuration Script
Sets up MLflow for experiment tracking with GCS backend.
"""

import os
import yaml
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


def load_mlflow_config():
    """Load MLflow configuration."""
    config_path = Path(__file__).parent.parent / "config" / "mlflow.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_mlflow(experiment_name: str = None) -> str:
    """
    Configure MLflow with GCS artifact store.
    Returns the experiment ID.
    """
    config = load_mlflow_config()

    # Set tracking URI
    tracking_uri = config['tracking']['tracking_uri']
    mlflow.set_tracking_uri(tracking_uri)

    # Get artifact location from config, allow env override
    artifact_location = os.environ.get(
        'MLFLOW_ARTIFACT_ROOT',
        config['tracking']['artifact_location']
    )

    # Use provided name or default
    if experiment_name is None:
        experiment_name = config['experiment']['experiments'][0]['name']

    # Create or get experiment
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_location
        )
        print(f"Created experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_test_experiment() -> str:
    """
    Log a test experiment to verify MLflow setup.
    Returns the run ID.
    """
    experiment_id = setup_mlflow("badminton-test")

    with mlflow.start_run(run_name="test-run") as run:
        # Log test parameters
        mlflow.log_param("test_param", "value")
        mlflow.log_param("model_type", "test")

        # Log test metrics
        mlflow.log_metric("accuracy", 0.75)
        mlflow.log_metric("f1_score", 0.72)

        # Log a simple artifact (text file)
        test_artifact = Path("test_artifact.txt")
        test_artifact.write_text("MLflow test artifact\n")
        mlflow.log_artifact(str(test_artifact))
        test_artifact.unlink()  # Clean up local file

        print(f"Logged test run: {run.info.run_id}")
        return run.info.run_id


def get_experiment_runs(experiment_name: str) -> list:
    """Get all runs for an experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    return runs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MLflow Configuration")
    parser.add_argument("--setup", type=str, help="Setup experiment by name")
    parser.add_argument("--test", action="store_true", help="Run test experiment")
    parser.add_argument("--list", type=str, help="List runs for experiment")
    args = parser.parse_args()

    if args.setup:
        exp_id = setup_mlflow(args.setup)
        print(f"Experiment ready: {exp_id}")
    elif args.test:
        run_id = log_test_experiment()
        print(f"\nTest complete. Run ID: {run_id}")
        print("Check mlruns/ directory or GCS bucket for artifacts")
    elif args.list:
        runs = get_experiment_runs(args.list)
        for run in runs[:10]:
            print(f"  {run.info.run_id}: {run.data.metrics}")
    else:
        parser.print_help()
