#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check for command line arguments
if [ "$1" == "cli" ]; then
    # Run the CLI interface
    python3 -m app.cli "${@:2}"
else
    # Run the web interface
    uvicorn app.main:app --host 0.0.0.0 --port 8000 
fi 