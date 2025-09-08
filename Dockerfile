FROM python:3.13-slim

# Install system dependencies for OpenCV and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables for Flask
ENV PYTHONUNBUFFERED=1

# Run the application with Gunicorn (production WSGI server)
# Run with unbuffered output so print statements appear in logs immediately
CMD ["python", "-u", "run.py"]