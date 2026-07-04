FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev libgomp1 git curl wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install FaceFusion dependencies + server deps
RUN pip install insightface onnxruntime-gpu opencv-python numpy aiohttp

# Clone FaceFusion 3.1.0
RUN git clone https://github.com/facefusion/facefusion.git /workspace/facefusion
WORKDIR /workspace/facefusion
RUN git checkout 3.1.0

# Install FaceFusion requirements & pre-download models
RUN pip install -r requirements.txt 2>/dev/null || true
RUN python facefusion.py force-download 2>/dev/null || true

# Clone our streaming server
RUN git clone https://github.com/okongor/faceswap.git /workspace/faceswap
WORKDIR /workspace/faceswap

# Set up PYTHONPATH
ENV PYTHONPATH=/workspace/facefusion:$PYTHONPATH

# Expose HTTP + WebSocket port
EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import socket; s=socket.socket(); s.settimeout(3); s.connect(('localhost',8888)); s.close()" || exit 1

CMD ["python", "-u", "/workspace/faceswap/stream_server.py"]
