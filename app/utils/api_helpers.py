"""
Utility functions for API interactions in the Streamlit app
"""
import os
import time
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

def get_api_base_url() -> str:
    """Get the API base URL from environment or default to localhost"""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


def format_api_url(base_url: str, endpoint: str = "api") -> str:
    """Format a complete API URL"""
    return f"{base_url.rstrip('/')}/{endpoint}"


def make_api_request(
    url: str, 
    question: Optional[str] = None, 
    image_file=None, 
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Make a request to the API endpoint
    
    Args:
        url: Complete API URL
        question: Optional question text
        image_file: Optional image file object
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with response data and metadata
    """
    files = {}
    data = {}
    
    if question:
        data["question"] = question

    if image_file:
        files["image"] = (
            image_file.name, 
            image_file.getvalue(), 
            image_file.type
        )

    try:
        start_time = time.time()
        
        if files:
            # Use multipart form for image uploads
            response = requests.post(url, data=data, files=files, timeout=timeout)
        else:
            # Use JSON for text-only questions
            response = requests.post(url, json={"question": question}, timeout=timeout)

        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            return {
                "answer": f"❌ Request failed: HTTP {response.status_code}",
                "links": [],
                "error": f"HTTP {response.status_code}",
                "elapsed": elapsed,
                "success": False
            }

        result = response.json()
        result["elapsed"] = elapsed
        result["success"] = True
        return result
        
    except requests.exceptions.Timeout:
        return {
            "answer": f"⏰ Request timed out after {timeout} seconds",
            "links": [],
            "error": "Timeout",
            "elapsed": timeout,
            "success": False
        }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "🔌 Connection error. Please check the API URL and ensure the backend is running.",
            "links": [],
            "error": "Connection Error",
            "elapsed": 0,
            "success": False
        }
    except requests.RequestException as e:
        return {
            "answer": f"❌ Network error: {str(e)}",
            "links": [],
            "error": str(e),
            "elapsed": 0,
            "success": False
        }


def check_api_health(api_url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Check the health of the API
    
    Args:
        api_url: Base API URL
        timeout: Health check timeout
        
    Returns:
        Dictionary with health status and information
    """
    try:
        health_url = f"{api_url.rstrip('/')}/"
        response = requests.get(health_url, timeout=timeout)
        
        if response.ok:
            try:
                data = response.json()
                return {
                    "healthy": True,
                    "status": "healthy",
                    "message": "API is healthy and responding",
                    "endpoints": data.get("endpoints", []),
                    "status_code": response.status_code
                }
            except:
                return {
                    "healthy": True,
                    "status": "healthy",
                    "message": "API is responding",
                    "endpoints": [],
                    "status_code": response.status_code
                }
        else:
            return {
                "healthy": False,
                "status": "unhealthy",
                "message": f"API responded with HTTP {response.status_code}",
                "endpoints": [],
                "status_code": response.status_code
            }
            
    except requests.exceptions.Timeout:
        return {
            "healthy": False,
            "status": "timeout",
            "message": "API health check timed out",
            "endpoints": [],
            "status_code": None
        }
    except requests.exceptions.ConnectionError:
        return {
            "healthy": False,
            "status": "connection_error",
            "message": "Cannot connect to API. Check the URL and ensure the backend is running.",
            "endpoints": [],
            "status_code": None
        }
    except Exception as e:
        return {
            "healthy": False,
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "endpoints": [],
            "status_code": None
        }
