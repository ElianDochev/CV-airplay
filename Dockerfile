FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        linux-libc-dev \
        libgl1 \
        libglib2.0-0 \
        libsdl2-2.0-0 \
        libsdl2-image-2.0-0 \
        libsdl2-ttf-2.0-0 \
        libsdl2-mixer-2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxtst6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD ["python", "-m", "src.app", "--no-show-ui"]
