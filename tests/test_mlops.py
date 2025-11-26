import sys
import os
import json
import time
import requests
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

# Mock server for webhooks
class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        print(f"\n[Webhook Received] {self.path}")
        print(f"Headers: {self.headers}")
        print(f"Body: {post_data.decode('utf-8')}")
        
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def start_webhook_server(port=8888):
    server = HTTPServer(('localhost', port), WebhookHandler)
    thread = Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    print(f"Webhook server started on port {port}")
    return server

def test_mlops():
    print("Testing MLOps Integration...")
    
    # Start webhook listener
    start_webhook_server()
    
    base_url = "http://localhost:8000"
    
    # 1. Register Webhook
    print("\n1. Registering Webhook...")
    webhook_data = {
        "url": "http://localhost:8888/webhook",
        "events": ["all"],
        "secret": "test_secret"
    }
    # Note: This assumes the API is running. If not, we can test the manager directly.
    # For this script, let's test the components directly to avoid needing the full API up.
    
    from mlops.webhooks import get_webhook_manager
    import database as db
    
    # Initialize DB for testing
    db.init_db()
    
    manager = get_webhook_manager()
    webhook_id = manager.register_webhook("http://localhost:8888/webhook", ["all"], "test_secret")
    print(f"Webhook registered with ID: {webhook_id}")
    
    # 2. Trigger Webhook
    print("\n2. Triggering Webhook...")
    manager.trigger_webhook("test_event", {"message": "Hello Webhook!"})
    time.sleep(1)  # Wait for async delivery if any (though manager is sync)
    
    # 3. Test Scheduler
    print("\n3. Testing Scheduler...")
    from mlops.scheduler import get_scheduler
    scheduler = get_scheduler()
    
    job_id = scheduler.schedule_job(
        name="Test Job",
        cron_expression="* * * * *",  # Every minute
        task_type="generation",
        parameters={"domain": "test", "count": 1}
    )
    print(f"Scheduled job created with ID: {job_id}")
    
    # 4. Test MLflow Logger (Mock)
    print("\n4. Testing MLflow Logger...")
    from mlops.mlflow_integration import get_mlflow_logger
    logger = get_mlflow_logger()
    # Just verify no errors when calling
    logger.log_generation(1, {"param": "value"}, {"metric": 0.95})
    print("MLflow logger called successfully")

if __name__ == "__main__":
    test_mlops()
