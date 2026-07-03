FROM runpod/base:0.7.3-cuda12.4.0

# System dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 libxrender-dev libgomp1 wget curl \
    && rm -rf /var/lib/apt/lists/*

# Python packages for real-time streaming
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# InsightFace for real-time face swap
RUN pip install --no-cache-dir insightface onnxruntime-gpu

# RVC for voice cloning (from GitHub)
RUN pip install --no-cache-dir git+https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git@main#subdirectory=rvc_python

# Copy server
COPY stream_server.py /app/stream_server.py
WORKDIR /app

# Pre-download FaceFusion models (for face detection)
RUN python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l')" 2>/dev/null || true

# Pre-download face swapper model
RUN python -c "from insightface.model_zoo import get_model; get_model('inswapper_128.onnx', download=True, download_zip=True)" 2>/dev/null || true

# Warm pod - keeps running, waiting for connections
EXPOSE 8765
CMD [ "python", "-u", "/app/stream_server.py" ]
