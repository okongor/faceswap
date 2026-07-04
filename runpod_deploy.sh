#!/bin/bash
# One-click RunPod pod deploy script
# Usage: bash runpod_deploy.sh <RUNPOD_API_KEY>

set -e

API_KEY="${1:-$RUNPOD_API_KEY}"
if [ -z "$API_KEY" ]; then
    echo "Usage: bash runpod_deploy.sh YOUR_RUNPOD_API_KEY"
    echo "Or set: export RUNPOD_API_KEY=your_key_here"
    exit 1
fi

echo "🚀 Deploying FaceSwap pod on RunPod..."
echo ""

DEPLOY_JSON=$(cat <<'EOF'
{
  "name": "faceswap-pro",
  "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "containerDiskSizeGb": 30,
  "ports": "8888/http",
  "gpuTypeIds": ["NVIDIA L4"],
  "gpuCount": 1,
  "startupScript": "#!/bin/bash\npip install -q opencv-python aiohttp numpy onnxruntime-gpu 2>/dev/null\ncd /workspace\nif [ ! -d facefusion ]; then git clone https://github.com/facefusion/facefusion.git && cd facefusion && git checkout 3.1.0 && pip install -q -r requirements.txt 2>/dev/null; fi\ncd /workspace\nif [ ! -d faceswap ]; then git clone https://github.com/okongor/faceswap.git; fi\ncd /workspace/faceswap && git pull\nmkdir -p /workspace/facefusion/.assets/models\ncurl -sL \"https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx\" -o /workspace/facefusion/.assets/models/inswapper_128.onnx\ncurl -sL \"https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface_10g.onnx\" -o /workspace/facefusion/.assets/models/retinaface_10g.onnx\ncurl -sL \"https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.onnx\" -o /workspace/facefusion/.assets/models/2dfan4.onnx\ncurl -sL \"https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.onnx\" -o /workspace/facefusion/.assets/models/arcface_w600k_r50.onnx\ncurl -sL \"https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fairface.onnx\" -o /workspace/facefusion/.assets/models/fairface.onnx\ncurl -sL \"https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/peppa_wutz.onnx\" -o /workspace/facefusion/.assets/models/peppa_wutz.onnx\ncurl -sL \"https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gender_age.onnx\" -o /workspace/facefusion/.assets/models/gender_age.onnx\ncurl -sL \"https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/xseg_1.onnx\" -o /workspace/facefusion/.assets/models/xseg_1.onnx\nexport PYTHONPATH=/workspace/facefusion:$PYTHONPATH\nexport STREAM_PORT=8888\nnohup python /workspace/faceswap/stream_server.py > /var/log/faceswap.log 2>&1 &\nsleep 3\ntail -5 /var/log/faceswap.log"
}
EOF
)

RESPONSE=$(curl -s -X POST "https://api.runpod.io/v2/pods" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "$DEPLOY_JSON")

POD_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)

if [ -n "$POD_ID" ]; then
    echo "✅ Pod deploying! ID: $POD_ID"
    echo ""
    echo "URL:  https://$POD_ID-8888.proxy.runpod.net"
    echo ""
    echo "Wait ~3 min for models to download, then open in Chrome."
    echo ""
    echo "Check status:"
    echo "  curl -s \"https://api.runpod.io/v2/pods/$POD_ID\" -H \"Authorization: Bearer $API_KEY\" | python3 -m json.tool"
else
    echo "❌ Deploy failed:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
fi
