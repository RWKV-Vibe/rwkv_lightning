# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir robyn pydantic ninja numpy

EXPOSE 8000

ENV MODEL_PATH=""
ENV PORT=8000
ENV PASSWORD=""
ENV RUNTIME=fp16

ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash", "-lc", "python app.py --model-path \"${MODEL_PATH}\" --runtime \"${RUNTIME}\" --port \"${PORT}\" ${PASSWORD:+--password \"${PASSWORD}\"}"]
