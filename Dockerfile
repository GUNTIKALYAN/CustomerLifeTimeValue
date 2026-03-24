# Use lightweight Python
FROM python:3.10-slim

# Set working dir
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1


# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]