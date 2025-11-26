import mlflow
import os
from typing import Dict, Any, Optional
from config import get_config

class MLflowLogger:
    def __init__(self, experiment_name: str = "dataset_creator"):
        self.config = get_config()
        self.enabled = self.config.get("mlops", {}).get("mlflow_enabled", False)
        
        if self.enabled:
            mlflow.set_experiment(experiment_name)

    def log_generation(self, job_id: int, params: Dict[str, Any], metrics: Dict[str, Any]):
        """Log generation job metrics"""
        if not self.enabled:
            return
            
        with mlflow.start_run(run_name=f"generation_job_{job_id}"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

    def log_quality_check(self, job_id: int, dataset_id: int, metrics: Dict[str, Any]):
        """Log quality check metrics"""
        if not self.enabled:
            return
            
        with mlflow.start_run(run_name=f"quality_check_{job_id}"):
            mlflow.log_param("dataset_id", dataset_id)
            mlflow.log_metrics(metrics)

    def log_dataset_artifact(self, file_path: str, artifact_path: str = "datasets"):
        """Log dataset file as artifact"""
        if not self.enabled:
            return
            
        with mlflow.start_run(run_name=f"dataset_upload"):
            mlflow.log_artifact(file_path, artifact_path)

# Global instance
_mlflow_logger = MLflowLogger()

def get_mlflow_logger():
    return _mlflow_logger
