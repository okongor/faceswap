# Real-time FaceSwap + Voice Clone

Live face swap (FaceFusion) + voice cloning (RVC) for 60-second video calls.

## How it works

**Single port.** HTTP (serves index.html) + WebSocket (at `/ws`) — all on one port. No extra port config needed.

## Deploy on RunPod

### Option A: Fresh pod (recommended)

1. **Deploy GPU Pod** → select **Runpod Pytorch 2.8.0** → pick **L4**
2. Set **Container Disk** to 30 GB
3. **Expose HTTP Ports**: just `8888` (default, Jupyter)
4. Deploy, wait for green status

5. **SSH into pod** and run:
```bash
# Install Python deps
pip install opencv-python aiohttp numpy

# Clone FaceFusion
cd /workspace
git clone https://github.com/facefusion/facefusion.git
cd facefusion
git checkout 3.1.0
pip install -r requirements.txt

# Clone faceswap repo
cd /workspace
git clone https://github.com/okongor/faceswap.git
cd faceswap
git pull

# Set PYTHONPATH
export PYTHONPATH=/workspace/facefusion:$PYTHONPATH
echo "export PYTHONPATH=/workspace/facefusion:\$PYTHONPATH" >> ~/.bashrc

# Start server
nohup python stream_server.py > /var/log/faceswap.log 2>&1 &

# Watch startup logs
tail -f /var/log/faceswap.log
```

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
