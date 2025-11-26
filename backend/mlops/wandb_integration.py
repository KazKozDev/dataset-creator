import wandb
import os
from typing import Dict, Any, Optional
from config import get_config

class WandbLogger:
    def __init__(self, project_name: str = "dataset_creator"):
        self.config = get_config()
        self.enabled = self.config.get("mlops", {}).get("wandb_enabled", False)
        self.project_name = project_name

    def log_generation(self, job_id: int, params: Dict[str, Any], metrics: Dict[str, Any]):
        """Log generation job metrics"""
        if not self.enabled:
            return
            
        wandb.init(project=self.project_name, name=f"generation_job_{job_id}", reinit=True)
        wandb.config.update(params)
        wandb.log(metrics)
        wandb.finish()

    def log_quality_check(self, job_id: int, dataset_id: int, metrics: Dict[str, Any]):
        """Log quality check metrics"""
        if not self.enabled:
            return
            
        wandb.init(project=self.project_name, name=f"quality_check_{job_id}", reinit=True)
        wandb.config.update({"dataset_id": dataset_id})
        wandb.log(metrics)
        wandb.finish()

# Global instance
_wandb_logger = WandbLogger()

def get_wandb_logger():
    return _wandb_logger
