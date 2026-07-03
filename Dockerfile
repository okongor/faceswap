FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev libgomp1 git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install insightface onnxruntime-gpu
RUN pip install websockets opencv-python numpy

RUN git clone https://github.com/facefusion/facefusion.git /facefusion
WORKDIR /facefusion
RUN git checkout 3.1.0
RUN pip install -r requirements.txt 2>/dev/null || true
RUN python -c "from facefusion.face_analyser import get_face_analyser; get_face_analyser()"
RUN python -c "from facefusion.processors.frame.modules.face_swapper import get_face_swapper; get_face_swapper()"

COPY stream_server.py /app/stream_server.py
WORKDIR /app

EXPOSE 8765
CMD ["python", "-u", "/app/stream_server.py"]
