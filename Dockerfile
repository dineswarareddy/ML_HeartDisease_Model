FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy source code and models
COPY src/ /app/src/
COPY models/ /app/models/

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]