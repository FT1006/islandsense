FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY config.yaml .

# Copy pre-built frontend
COPY frontend/dist/ ./frontend/dist/

# Expose port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "src.islandsense.api:app", "--host", "0.0.0.0", "--port", "8000"]
