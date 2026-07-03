# Real-time FaceSwap + Voice Clone

Live face swap (FaceFusion) + voice cloning (RVC) for 60-second video calls.

## Quick Deploy (RunPod)

1. Go to **Pod Settings / Edit** on your running pod
2. Add Exposed Port: **8765 TCP** (or add it when deploying)
3. SSH into pod and run:
```bash
cd /workspace && git clone https://github.com/okongor/faceswap.git && cd faceswap
export PYTHONPATH=/workspace/facefusion:$PYTHONPATH
echo "export PYTHONPATH=/workspace/facefusion:\$PYTHONPATH" >> ~/.bashrc
pip install websockets
nohup python stream_server.py > /var/log/faceswap.log 2>&1 &
```
4. Open `index.html` in browser, enter `ws://<pod-ip>:8765`

## Files

- `stream_server.py` — WebSocket server (FaceFusion + RVC)
- `api_server.py` — Queue/quota API (FastAPI)
- `index.html` — Browser client
- `Dockerfile` — Builds container with everything pre-installed

## Docker Build (alternative)

```bash
docker build -t yourname/faceswap-stream .
docker push yourname/faceswap-stream
```
Then use `yourname/faceswap-stream` as container image on RunPod.
