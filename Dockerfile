FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV API_BASE_URL=https://api.openai.com/v1
ENV GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
ENV MODEL_NAME=gemini-3-flash-preview
ENV INFERENCE_BACKEND=gemini

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
