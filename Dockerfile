FROM runpod/base:0.7.3-cuda12.4.0

# Install system dependencies for FaceFusion
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone FaceFusion
WORKDIR /
RUN git clone https://github.com/facefusion/facefusion.git

WORKDIR /facefusion
RUN git checkout 3.1.0  # Pin to stable release

# Install FaceFusion with CUDA
RUN python install.py --torch cuda --onnxruntime cuda

# Install RunPod SDK and boto3 (for S3 output)
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy our handler
COPY src/handler.py /facefusion/handler.py

# Pre-download FaceFusion models at build time
RUN python facefusion.py force-download || true

# RunPod serverless entrypoint
CMD [ "python", "-u", "/facefusion/handler.py" ]
