#!/usr/bin/env python
import os
import sys
import logging
import subprocess
import time
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_grpc_server():
    """Start the gRPC tools server in a subprocess"""
    logger.info("Starting gRPC tools server...")
    grpc_server_script = Path(__file__).resolve().parent.parent / "app" / "tools_engine" / "server.py"
    
    # Check if the file exists
    if not grpc_server_script.exists():
        logger.error(f"gRPC server script not found at {grpc_server_script}")
        return None
    
    # Start the gRPC server as a subprocess
    process = subprocess.Popen(
        [sys.executable, str(grpc_server_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    logger.info(f"gRPC tools server started (PID: {process.pid})")
    return process

def start_fastapi_server():
    """Start the FastAPI server"""
    logger.info("Starting FastAPI server...")
    
    # Get the path to the main.py file
    main_script = Path(__file__).resolve().parent.parent / "app" / "main.py"
    
    # Check if the file exists
    if not main_script.exists():
        logger.error(f"Main application script not found at {main_script}")
        return None
    
    # Start the FastAPI server with uvicorn
    process = subprocess.Popen(
        [
            "uvicorn", 
            "app.main:app", 
            "--host", settings.API_HOST, 
            "--port", str(settings.API_PORT),
            "--reload" if settings.DEBUG else ""
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    logger.info(f"FastAPI server started (PID: {process.pid})")
    return process

def monitor_processes(processes):
    """Monitor running processes and log their output"""
    try:
        while all(p.poll() is None for p in processes):
            for p in processes:
                # Check stdout
                stdout_line = p.stdout.readline()
                if stdout_line:
                    print(stdout_line.strip())
                
                # Check stderr
                stderr_line = p.stderr.readline()
                if stderr_line:
                    print(stderr_line.strip(), file=sys.stderr)
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping servers...")
        for p in processes:
            if p.poll() is None:  # If process is still running
                p.terminate()
                p.wait()
        logger.info("All servers stopped.")

if __name__ == "__main__":
    logger.info("Starting Call Center Information System...")
    
    # Start the gRPC server
    grpc_process = start_grpc_server()
    
    # Wait a moment for the gRPC server to start
    time.sleep(2)
    
    # Start the FastAPI server
    api_process = start_fastapi_server()
    
    # Monitor both processes
    processes = [p for p in [grpc_process, api_process] if p is not None]
    if processes:
        logger.info("All servers started. Press Ctrl+C to stop.")
        monitor_processes(processes)
    else:
        logger.error("Failed to start servers.") 