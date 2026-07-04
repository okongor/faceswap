# 🎭 FaceSwap Pro

Real-time face swap streaming server using FaceFusion + ONNX GPU.
Single port: HTTP (index.html) + WebSocket (/ws).

## 🚀 Deploy on RunPod (One Click)

Run this from any terminal:
```bash
bash <(curl -s https://raw.githubusercontent.com/okongor/faceswap/main/runpod_deploy.sh) YOUR_RUNPOD_API_KEY
```

Or manually:

### 1. Deploy GPU Pod
- **Template**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- **GPU**: NVIDIA L4 (or any)
- **Container Disk**: 30 GB
- **Exposed Ports**: `8888/http`
- **Startup Script**:
```bash
pip install -q opencv-python aiohttp numpy onnxruntime-gpu 2>/dev/null
cd /workspace
if [ ! -d facefusion ]; then git clone https://github.com/facefusion/facefusion.git && cd facefusion && git checkout 3.1.0 && pip install -q -r requirements.txt 2>/dev/null; fi
cd /workspace
if [ ! -d faceswap ]; then git clone https://github.com/okongor/faceswap.git; fi
cd /workspace/faceswap && git pull
export PYTHONPATH=/workspace/facefusion:$PYTHONPATH
export STREAM_PORT=8888
nohup python /workspace/faceswap/stream_server.py > /var/log/faceswap.log 2>&1 &
sleep 3
tail -5 /var/log/faceswap.log
```

### 2. Open in Browser
**`https://[POD_ID]-8888.proxy.runpod.net`**

### 3. Use It
1. Connect
2. Upload a source face photo
3. Toggle "Preserve skin & hair" (on by default — keeps target's skin tone and hair)
4. Start Call

The server stays alive as long as the pod is running. Even if you close the terminal.

6. Once you see `"Server on http://0.0.0.0:8888"` — open in browser:

   **`https://[POD_ID]-8888.proxy.runpod.net`**

### Option B: Using a different port

If port 8888 is already taken by Jupyter:

1. Edit pod → **Expose HTTP Ports** → add `8890,8891`
2. Run with: `export STREAM_PORT=8890` (or 8891)
3. Open: **`https://[POD_ID]-8890.proxy.runpod.net`**

### Option C: Docker build

```bash
docker build -t yourname/faceswap-stream .
docker push yourname/faceswap-stream
```
Use as container image on RunPod.
