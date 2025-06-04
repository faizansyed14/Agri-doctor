# Stage 1: React Build
FROM node:18 AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# Stage 2: Python + FastAPI
FROM python:3.9-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY api ./api

# Copy frontend build output
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Expose port
ENV PYTHONUNBUFFERED=1
EXPOSE $PORT

# Start app with Gunicorn
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT api.main:app
