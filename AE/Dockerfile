FROM python:3.13-slim

# Install system dependencies for OpenCV and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /AE

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5005

# Set environment variables for Flask
ENV PYTHONUNBUFFERED=1
ENV MAX_CONTENT_LENGTH=1073741824
ENV MAX_FORM_PARTS=500000
ENV MAX_FORM_MEMORY_SIZE=524288000
ENV FLASK_RUN_HOST=0.0.0.0

# Flask configuration
ENV FLASK_DEBUG=True

# MLFLow configuration
ENV MLFLOW_TRACKING_URI="https://mlflow.simplex4learning.de"

# Run the application
# CMD ["python", "-u", "run_extension.py"]


ENTRYPOINT ["python3"]

CMD ["-m", "gunicorn", \
     "--preload", \
     "--bind", "0.0.0.0:5005", \
     "--threads", "4", \
     "--log-level", "debug", \
     "run_extension:analytics_service()"]
