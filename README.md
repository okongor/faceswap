# FaceFusion on RunPod Serverless

Run FaceFusion as a serverless endpoint on RunPod.

## What it does

Accepts a source face image + a target image/video, runs face swapping (and optionally face enhancement), returns the output file URL over S3/R2.

## Deploy

1. Fork this repo
2. Go to **RunPod Console → Serverless → Deploy from a GitHub repository**
3. Connect this repo
4. Set your endpoint config (GPU type, concurrency, etc.)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `S3_ENDPOINT` | Yes | S3-compatible storage endpoint |
| `S3_REGION` | Yes | S3 region (e.g. `us-east-1`) |
| `S3_BUCKET` | Yes | Bucket name for output |
| `S3_ACCESS_KEY` | Yes | S3 access key |
| `S3_SECRET_KEY` | Yes | S3 secret key |
| `OUTPUT_PATH_PREFIX` | No | Prefix for output keys (default: `facefusion/`) |

## Input Format

```json
{
  "input": {
    "source_face_url": "https://.../source.jpg",
    "target_url": "https://.../target.mp4",
    "processors": ["face_swapper", "face_enhancer"],
    "execution_providers": ["cuda"],
    "output_format": "mp4"
  }
}
```

## Output Format

```json
{
  "output_url": "https://s3.../facefusion/output_abc123.mp4"
}
```
