# CPU test image. Triton 3.x ships Linux wheels only, so kernels are tested here through
# the Triton interpreter and compiled for sm_89 without a GPU.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ make git rsync openssh-client procps \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch transformers safetensors numpy matplotlib triton \
    fastapi uvicorn httpx pytest hypothesis huggingface_hub openai
WORKDIR /src
