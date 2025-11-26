from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
import json
from datetime import datetime
import database as db
from mlops.celery_app import run_generation_job, run_quality_check_job
from config import get_config

logger = logging.getLogger(__name__)

class JobScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

    def schedule_job(self, name: str, cron_expression: str, task_type: str, parameters: dict):
        """Schedule a new periodic job"""
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO scheduled_jobs (name, cron_expression, task_type, parameters, active, created_at)
        VALUES (?, ?, ?, ?, 1, ?)
        ''', (name, cron_expression, task_type, json.dumps(parameters), now))
        
        job_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Add to APScheduler
        self._add_to_scheduler(job_id, name, cron_expression, task_type, parameters)
        
        return job_id

    def _add_to_scheduler(self, db_job_id: int, name: str, cron_expression: str, task_type: str, parameters: dict):
        """Internal method to add job to APScheduler"""
        trigger = CronTrigger.from_crontab(cron_expression)
        
        self.scheduler.add_job(
            self._execute_job,
            trigger,
            args=[db_job_id, task_type, parameters],
            id=str(db_job_id),
            name=name,
            replace_existing=True
        )

    def _execute_job(self, db_job_id: int, task_type: str, parameters: dict):
        """Execute the scheduled job"""
        logger.info(f"Executing scheduled job {db_job_id} ({task_type})")
        
        # Update last run time
        conn = db.get_db_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute('UPDATE scheduled_jobs SET last_run_at = ? WHERE id = ?', (now, db_job_id))
        conn.commit()
        conn.close()
        
        # Trigger the actual task via Celery
        config = get_config()
        provider_config = {
            "provider": parameters.get("provider", config.get_default_provider()),
            "model": parameters.get("model"),
        }
        
        if task_type == "generation":
            # Create a generation job record first
            gen_job_id = db.create_generation_job(
                examples_requested=parameters.get("count", 10),
                parameters=parameters
            )
            run_generation_job.delay(gen_job_id, parameters, provider_config)
            
        elif task_type == "quality_check":
            # Create quality job record
            # Note: This assumes dataset_id is in parameters
            if "dataset_id" in parameters:
                # We need to calculate total examples, which is tricky here without DB access
                # For now, we'll just pass 0 and let the task update it
                qual_job_id = db.create_quality_job(
                    dataset_id=parameters["dataset_id"],
                    examples_total=0, 
                    parameters=parameters
                )
                run_quality_check_job.delay(qual_job_id, parameters, provider_config)

    def load_jobs_from_db(self):
        """Load all active jobs from database on startup"""
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM scheduled_jobs WHERE active = 1')
        jobs = cursor.fetchall()
        conn.close()
        
        for job in jobs:
            try:
                self._add_to_scheduler(
                    job['id'],
                    job['name'],
                    job['cron_expression'],
                    job['task_type'],
                    json.loads(job['parameters'])
                )
            except Exception as e:
                logger.error(f"Failed to load job {job['id']}: {e}")

# Global instance
_scheduler = JobScheduler()

def get_scheduler():
    return _scheduler
