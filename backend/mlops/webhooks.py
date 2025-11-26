import json
import hmac
import hashlib
import requests
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import database as db

logger = logging.getLogger(__name__)

class WebhookManager:
    def __init__(self):
        pass

    def register_webhook(self, url: str, events: List[str], secret: Optional[str] = None) -> int:
        """Register a new webhook"""
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO webhooks (url, events, secret, created_at, active)
        VALUES (?, ?, ?, ?, 1)
        ''', (url, json.dumps(events), secret, now))
        
        webhook_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return webhook_id

    def list_webhooks(self) -> List[Dict[str, Any]]:
        """List all webhooks"""
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM webhooks WHERE active = 1')
        webhooks = cursor.fetchall()
        conn.close()
        
        return [dict(w) for w in webhooks]

    def trigger_webhook(self, event: str, payload: Dict[str, Any]):
        """Trigger webhooks for a specific event"""
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        # Find webhooks subscribed to this event
        cursor.execute('SELECT * FROM webhooks WHERE active = 1')
        webhooks = cursor.fetchall()
        
        for webhook in webhooks:
            events = json.loads(webhook['events'])
            if event in events or 'all' in events:
                self._send_webhook(webhook, event, payload)
        
        conn.close()

    def _send_webhook(self, webhook: Dict[str, Any], event: str, payload: Dict[str, Any]):
        """Send a single webhook request"""
        url = webhook['url']
        secret = webhook['secret']
        
        headers = {
            'Content-Type': 'application/json',
            'X-Event-Type': event,
            'User-Agent': 'LLM-Dataset-Creator/1.0'
        }
        
        # Calculate signature if secret is present
        body = json.dumps(payload)
        if secret:
            signature = hmac.new(
                secret.encode('utf-8'),
                body.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            headers['X-Hub-Signature-256'] = f"sha256={signature}"
        
        try:
            response = requests.post(url, data=body, headers=headers, timeout=10)
            status_code = response.status_code
            response_body = response.text[:1000]  # Truncate if too long
        except Exception as e:
            logger.error(f"Failed to send webhook to {url}: {e}")
            status_code = 0
            response_body = str(e)
            
        # Log the attempt
        self._log_attempt(webhook['id'], event, payload, status_code, response_body)

    def _log_attempt(self, webhook_id: int, event: str, payload: Dict[str, Any], status_code: int, response_body: str):
        """Log webhook attempt to database"""
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO webhook_logs (webhook_id, event, payload, status_code, response_body, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (webhook_id, event, json.dumps(payload), status_code, response_body, now))
        
        conn.commit()
        conn.close()

# Global instance
_webhook_manager = WebhookManager()

def get_webhook_manager():
    return _webhook_manager
