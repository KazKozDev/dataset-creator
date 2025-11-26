import os
from celery import Celery
from config import get_config
import generator
import quality
from mlops.webhooks import get_webhook_manager

# Get configuration
config = get_config()
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery
celery_app = Celery(
    "dataset_creator",
    broker=redis_url,
    backend=redis_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

@celery_app.task(bind=True)
def run_generation_job(self, job_id: int, params: dict, provider_config: dict):
    """Run a generation job in the background"""
    try:
        # Reconstruct provider from config
        from llm_providers import get_provider
        provider = get_provider(
            provider_config.get("provider"),
            model=provider_config.get("model"),
            api_key=provider_config.get("api_key"),
            base_url=provider_config.get("base_url")
        )
        
        # Run generation
        generator.start_generation_job(job_id, params, provider)
        
        # Trigger webhook
        get_webhook_manager().trigger_webhook("generation_completed", {"job_id": job_id, "status": "success"})
        
    except Exception as e:
        # Trigger failure webhook
        get_webhook_manager().trigger_webhook("generation_failed", {"job_id": job_id, "error": str(e)})
        raise e

@celery_app.task(bind=True)
def run_quality_check_job(self, job_id: int, params: dict, provider_config: dict):
    """Run a quality check job in the background"""
    try:
        # Reconstruct provider
        from llm_providers import get_provider
        provider = get_provider(
            provider_config.get("provider"),
            model=provider_config.get("model"),
            api_key=provider_config.get("api_key"),
            base_url=provider_config.get("base_url")
        )
        
        # Run quality check
        quality.process_quality_job(
            job_id,
            provider,
            params.get("batch_size", 10),
            params.get("threshold", 7.0),
            params.get("auto_fix", False),
            params.get("auto_remove", False)
        )
        
        # Trigger webhook
        get_webhook_manager().trigger_webhook("quality_check_completed", {"job_id": job_id, "status": "success"})
        
    except Exception as e:
        get_webhook_manager().trigger_webhook("quality_check_failed", {"job_id": job_id, "error": str(e)})
        raise e
