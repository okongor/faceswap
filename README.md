# Real-time FaceSwap + Voice Clone

Live face swap and voice cloning for 60-second video calls.

- **stream_server.py** - WebSocket server, runs on GPU pod
- **api_server.py** - Queue/quota API, runs on CPU
- **test_client.html** - Browser test client
- **Dockerfile** - GPU pod container

## Deploy GPU Pod

1. RunPod -> GPU Pod -> Deploy from this repo
2. Expose port 8765 TCP
3. Set env: `MAX_USERS=1`, `CALL_DURATION=65`
