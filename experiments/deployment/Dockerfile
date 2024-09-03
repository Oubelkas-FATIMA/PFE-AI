FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py /app/
COPY "PATH_TO_FILE_MODEL_TO_DEPLOY" /app/

EXPOSE 5000

CMD ["python", "app.py"]
