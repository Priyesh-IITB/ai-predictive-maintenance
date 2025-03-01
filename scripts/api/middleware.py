"""
Middleware for the API to track metrics and performance.
"""
import time
from fastapi import Request
from datetime import datetime
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize metrics tracking
request_count = 0
error_count = 0
response_times = []
risk_levels = defaultdict(int)

async def monitor_requests(request: Request, call_next):
    """
    Middleware to monitor and collect metrics on API requests.
    """
    global request_count, error_count
    
    # Track request
    request_count += 1
    
    # Track performance
    start_time = time.time()
    
    try:
        # Execute the request
        response = await call_next(request)
        
        # Record successful response
        response_time = (time.time() - start_time) * 1000  # in milliseconds
        response_times.append(response_time)
        
        # Log request
        logger.debug(f"Request: {request.method} {request.url.path} - {response.status_code}")
        
        return response
    except Exception as e:
        # Track errors
        error_count += 1
        logger.error(f"Error in request: {str(e)}")
        raise e

def record_prediction(valve_id: int, risk_value: float, confidence: float, risk_level: str):
    """
    Record prediction statistics for monitoring and alerting.
    """
    # Track risk level distribution
    risk_levels[risk_level] += 1
    
    # Log prediction
    logger.debug(f"Prediction for valve {valve_id}: {risk_level} ({risk_value:.1%})")

def get_metrics():
    """
    Return current API metrics.
    """
    global request_count, error_count, response_times, risk_levels
    
    # Calculate metrics
    avg_response_time = sum(response_times) / max(len(response_times), 1)
    error_rate = error_count / max(request_count, 1)
    
    # Return metrics
    metrics = {
        "request_count": request_count,
        "error_count": error_count,
        "average_response_time": avg_response_time,
        "error_rate": error_rate,
        "risk_levels": dict(risk_levels),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return metrics