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

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_DEBUG=True
ENV VISUALISATION_HOST="http://127.0.0.1:5000"

# Counteract Status Code 413 in Flask
ENV MAX_CONTENT_LENGTH=1073741824
ENV MAX_FORM_PARTS=500000
ENV MAX_FORM_MEMORY_SIZE=524288000

# MLflow config
ENV RUN_ID="0d7a12e7865d40bd85bb431d1f990065"
ENV MLFLOW_TRACKING_TOKEN="eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJqNUpOUlhIX0JWZUZQS1lRanFUZ3lyWi1rYVJoUmJybGtPOWlmZjdDSDJvIn0.eyJleHAiOjE3NTg4MTk1MTksImlhdCI6MTc1ODc4MzUxOSwianRpIjoiNjRkZWM3YTUtY2ZjOC00OTUwLWE0YzktNTNlMTUyMjcwNTk1IiwiaXNzIjoiaHR0cHM6Ly9hdXRoLnNpbXBsZXg0bGVhcm5pbmcuZGUvcmVhbG1zL3NpbXBsZXg0bGVhcm5pbmciLCJhdWQiOlsibWxmbG93IiwiYWNjb3VudCJdLCJzdWIiOiJjN2U2YTU2NC0xZDAyLTQxZGEtOWRlOC0xNTA2ZTc0YzM5YjUiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJtbGZsb3ctYXBpIiwic2Vzc2lvbl9zdGF0ZSI6IjU1MzQ1NmYzLWQ5MTAtNDYzMC1hNGZmLTc1NmU2YTcyYzU4ZSIsImFjciI6IjEiLCJhbGxvd2VkLW9yaWdpbnMiOlsiLyoiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNpbXBsZXg0bGVhcm5pbmciXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIiwic2lkIjoiNTUzNDU2ZjMtZDkxMC00NjMwLWE0ZmYtNzU2ZTZhNzJjNThlIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJNYXJpdXMgSGVycm1hbm4iLCJncm91cHMiOlsiQWRtaW5pc3RyYXRvciIsIkFuYWx5c3QiLCJDcmVhdG9yIiwiVmlld2VyIl0sInByZWZlcnJlZF91c2VybmFtZSI6Im1hcml1cyIsImdpdmVuX25hbWUiOiJNYXJpdXMiLCJmYW1pbHlfbmFtZSI6IkhlcnJtYW5uIiwiZW1haWwiOiJtYXJpdXMuaGVycm1hbm5AZGlzeS5uZXQifQ.S70Zq1fe7DnO72Xuwm0eZvfkdkjNrjzUUW_dOTUg6AJ-mm6xEW8QgnAHW_z3afYs401ky8OgLcw1pXe5LQE-c2YjwYHQD7MklVM-M_BJdFDxllJnQaW7qqHsJ9CdtxibpKBMY9uq6Ej0-pwDB2iQETZ7ZSA25Ovc9BmAT1HokYzs4UCZL1uFgefuSMMb-l73CbVjw6DnMsVKiBfaKWZI4D0oPhPhFKKYta32hVo9NdY3tmRy1B8Id9UhXwLtJ5UVa2_JiD_Fm-6cCLY8Zn8oL8hksTI3WTEGxCwR-ZVdxkR0CNrTRdUE3SmaITJhQMUy3G__Qtm74D9Z8DiFQ3U7-w"
ENV MLFLOW_TRACKING_URI="https://mlflow.simplex4learning.de"


ENTRYPOINT ["python3"]


CMD ["-m", "gunicorn", \
     "--preload", \
     "--bind", "0.0.0.0:5005", \
     # Logging
     "--log-level", "debug", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output", \
     # Counteract Status Code 413 in Gunicorn
     "--limit-request-line", "16384", \
     "--limit-request-fields", "32768", \
     "--limit-request-field_size", "65536", \
     "--timeout", "1000", \
     "--worker-class", "gevent", \
     "--workers", "2", \
     "--threads", "4", \
     # Launch
     "run_extension:app"]