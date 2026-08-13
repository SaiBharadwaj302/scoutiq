FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies — lean runtime subset only (see requirements-api.txt
# for why this isn't the full requirements.txt used for training/CI).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source
COPY . .

EXPOSE 8000

CMD ["python", "run_api.py"]
