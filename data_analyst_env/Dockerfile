FROM python:3.10-slim
WORKDIR /code
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]