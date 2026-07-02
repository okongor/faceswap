"""
FaceFusion RunPod Serverless Handler

Receives a job with source face URL + target URL, runs FaceFusion
headless-run, uploads the output to RunPod's temp storage, and
returns the download URL.
"""
import os
import sys
import uuid
import subprocess
import tempfile
import shutil
from pathlib import Path

import runpod
import requests

FACEFUSION_SCRIPT = Path("/facefusion/facefusion.py")


def download_file(url: str, dest: Path) -> Path:
    """Download a file from URL to dest path."""
    print(f"Downloading {url} -> {dest}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {dest} ({dest.stat().st_size} bytes)")
    return dest


def run_facefusion(
    source_path: Path,
    target_path: Path,
    output_path: Path,
    processors: list[str],
    execution_providers: list[str],
) -> None:
    """Run facefusion headless-run."""
    cmd = [
        sys.executable,
        str(FACEFUSION_SCRIPT),
        "headless-run",
        "--source-paths", str(source_path),
        "--target-path", str(target_path),
        "--output-path", str(output_path),
        "--processors", *processors,
        "--execution-providers", *execution_providers,
        "--execution-thread-count", "4",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"FaceFusion stderr:\n{result.stderr}")
        raise RuntimeError(f"FaceFusion failed (rc={result.returncode}): {result.stderr[:2000]}")

    if result.stdout:
        print(f"FaceFusion stdout:\n{result.stdout}")

    if not output_path.exists():
        raise RuntimeError(f"Output file not found at {output_path}")


def handler(job):
    """RunPod handler - called for each job."""
    job_input = job["input"]
    job_id = job.get("id", str(uuid.uuid4()))

    source_face_url = job_input.get("source_face_url")
    target_url = job_input.get("target_url")
    processors = job_input.get("processors", ["face_swapper"])
    execution_providers = job_input.get("execution_providers", ["cuda"])
    output_format = job_input.get("output_format")

    if not source_face_url or not target_url:
        raise ValueError("Both 'source_face_url' and 'target_url' are required")

    work_dir = Path(tempfile.mkdtemp(prefix=f"facefusion_{job_id}_"))

    try:
        source_ext = Path(source_face_url.split("?")[0]).suffix or ".jpg"
        target_ext = Path(target_url.split("?")[0]).suffix or ".mp4"

        source_path = work_dir / f"source{source_ext}"
        target_path = work_dir / f"target{target_ext}"

        download_file(source_face_url, source_path)
        download_file(target_url, target_path)

        if output_format:
            output_ext = f".{output_format.lstrip('.')}"
        else:
            output_ext = target_ext

        output_path = work_dir / f"output_{job_id[:8]}{output_ext}"

        run_facefusion(source_path, target_path, output_path, processors, execution_providers)

        # Upload via RunPod - returns a temp download URL
        with open(output_path, "rb") as f:
            upload_response = runpod.api.upload_file(file_content=f, file_name=output_path.name)

        print(f"Upload response: {upload_response}")

        # The upload URL is usually in the response
        # Adjust the key based on RunPod's actual response format
        if isinstance(upload_response, dict):
            output_url = upload_response.get("url") or upload_response.get("downloadUrl") or upload_response.get("file_url")
        else:
            output_url = str(upload_response)

        if not output_url:
            raise RuntimeError(f"Failed to get upload URL from RunPod: {upload_response}")

        return {"output_url": output_url}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
