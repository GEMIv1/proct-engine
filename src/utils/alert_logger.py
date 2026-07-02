import os
from datetime import datetime

from models import AppConfig

class AlertLogger:
    def __init__(self, config: AppConfig):
        self.log_path = config.logging.log_path
        self.alerts = []
        self.cooldown = config.logging.alert_cooldown
        self.last_alert_time = {}
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_path, exist_ok=True)
        
    def log_alert(self, alert_type, message):
        """Log an alert with type and message"""
        current_time = datetime.now().timestamp()
        
        # Normalise to string for cooldown tracking (supports both str and enum)
        alert_key = str(alert_type)
        
        # Check cooldown for this alert type
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.cooldown:
                return None
                
        self.last_alert_time[alert_key] = current_time
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {alert_key.upper()}: {message}"
        self.alerts.append(log_entry)
        
        # Save to file
        log_file = os.path.join(self.log_path, "alerts.log")
        with open(log_file, "a") as f:
            f.write(log_entry + "\n")
            
        return log_entry