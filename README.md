# Real-time FaceSwap + Voice Clone

Live face swap (FaceFusion) + voice cloning (RVC) for 60-second video calls.

## Deploy on RunPod (No Extra Port Config Needed)

This server runs on **port 8888** — the same port RunPod's official PyTorch templates already expose for JupyterLab. No extra port configuration needed.

### Method 1: Deploy with RunPod PyTorch Template (Easiest)

1. Go to **Deploy GPU Pod** → select **Runpod Pytorch 2.8.0** → pick **L4** ($0.39/hr)
2. On the deploy page, set **Container Disk** to at least `30 GB`
3. In **Expose HTTP Ports**, it should already show `8888` (default Jupyter port)
4. Deploy the pod

5. Once running, **SSH** into pod and run:
```bash
cd /workspace && \
git clone https://github.com/okongor/faceswap.git && \
cd faceswap

# Set up FaceFusion path
export PYTHONPATH=/workspace/facefusion:$PYTHONPATH
echo "export PYTHONPATH=/workspace/facefusion:\$PYTHONPATH" >> ~/.bashrc

# Install deps
pip install websockets

# Start the server (runs on port 8888)
nohup python stream_server.py > /var/log/faceswap.log 2>&1 &
```

6. Open `index.html` in browser, enter:
   ```
   wss://[YOUR-POD-ID]-8888.proxy.runpod.net
   ```

### Method 2: Docker Build + Deploy

```bash
docker build -t yourname/faceswap-stream .
docker push yourname/faceswap-stream
```
Then use `yourname/faceswap-stream` as container image on RunPod.

## Files

- `stream_server.py` — WebSocket server (FaceFusion + RVC, port 8888 default)
- `index.html` — Browser client
- `Dockerfile` — Full container with FaceFusion + models pre-loaded
