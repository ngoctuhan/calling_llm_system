FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for vector databases and graph databases
RUN pip install --no-cache-dir \
    qdrant-client \
    neo4j \
    sentence-transformers

# Copy the application code
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Run the application with proper config for containerized environment
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 