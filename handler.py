#!/usr/bin/env python3
"""
RunPod Serverless Handler for Hallo2

Audio-driven portrait animation - takes a portrait image and audio,
generates a lip-synced talking head video.

Input format:
{
    "input": {
        "source_image": "base64 or URL",
        "driving_audio": "base64 or URL",
        "pose_weight": 1.0,
        "face_weight": 1.0,
        "lip_weight": 1.0,
        "face_expand_ratio": 1.2,
        "cfg_scale": 3.5,
        "steps": 40
    }
}

Output format:
{
    "video_url": "S3 URL to generated video" or
    "video_base64": "base64 encoded video",
    "duration_seconds": 15.2,
    "resolution": "512x512"
}
"""

import os
import sys
import json
import base64
import tempfile
import time
import urllib.request
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# Add Hallo2 to path
sys.path.insert(0, '/app/hallo2')

import runpod

# Configuration
HALLO2_DIR = Path("/app/hallo2")
MODELS_DIR = HALLO2_DIR / "pretrained_models"
INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, output_path: str) -> str:
    """Download a file from URL."""
    print(f"Downloading {url} to {output_path}")
    urllib.request.urlretrieve(url, output_path)
    return output_path


def save_base64_file(data: str, output_path: str) -> str:
    """Save base64 encoded data to a file."""
    # Remove data URI prefix if present
    if ',' in data:
        data = data.split(',', 1)[1]

    binary_data = base64.b64decode(data)
    with open(output_path, 'wb') as f:
        f.write(binary_data)
    return output_path


def file_to_base64(file_path: str) -> str:
    """Convert a file to base64 string."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def run_hallo2_inference(
    source_image_path: str,
    driving_audio_path: str,
    output_path: str,
    pose_weight: float = 1.0,
    face_weight: float = 1.0,
    lip_weight: float = 1.0,
    face_expand_ratio: float = 1.2,
    cfg_scale: float = 3.5,
    steps: int = 40
) -> str:
    """
    Run Hallo2 inference using the command-line interface.

    Returns the path to the generated video.
    """
    # Create config file for this inference
    config = {
        "source_image": source_image_path,
        "driving_audio": driving_audio_path,
        "output": output_path,
        "pose_weight": pose_weight,
        "face_weight": face_weight,
        "lip_weight": lip_weight,
        "face_expand_ratio": face_expand_ratio,
        "cfg_scale": cfg_scale,
        "steps": steps
    }

    config_path = INPUT_DIR / "inference_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)

    print(f"Running Hallo2 inference with config: {config}")

    # Run inference script
    # The exact command depends on Hallo2's CLI interface
    cmd = [
        "python", str(HALLO2_DIR / "scripts" / "inference_long.py"),
        "--config", str(HALLO2_DIR / "configs" / "inference" / "long.yaml"),
        "--source_image", source_image_path,
        "--driving_audio", driving_audio_path,
        "--output", output_path,
        "--pose_weight", str(pose_weight),
        "--face_weight", str(face_weight),
        "--lip_weight", str(lip_weight),
        "--face_expand_ratio", str(face_expand_ratio),
    ]

    print(f"Executing: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(HALLO2_DIR),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        if result.returncode != 0:
            raise Exception(f"Inference failed with code {result.returncode}: {result.stderr}")

    except subprocess.TimeoutExpired:
        raise Exception("Inference timed out after 10 minutes")

    # Find the output video
    if os.path.exists(output_path):
        return output_path

    # Try to find output in default location
    output_files = list(OUTPUT_DIR.glob("*.mp4"))
    if output_files:
        return str(max(output_files, key=os.path.getctime))

    raise Exception("No output video generated")


def run_hallo2_python(
    source_image_path: str,
    driving_audio_path: str,
    output_path: str,
    **kwargs
) -> str:
    """
    Run Hallo2 inference using Python API directly.

    This is an alternative to the CLI approach that may be more reliable.
    """
    try:
        # Try to import Hallo2 modules
        from hallo.animate import HalloAnimator
        from omegaconf import OmegaConf

        # Load config
        config_path = HALLO2_DIR / "configs" / "inference" / "long.yaml"
        config = OmegaConf.load(str(config_path))

        # Override config with input parameters
        config.source_image = source_image_path
        config.driving_audio = driving_audio_path
        config.output = output_path
        config.pose_weight = kwargs.get('pose_weight', 1.0)
        config.face_weight = kwargs.get('face_weight', 1.0)
        config.lip_weight = kwargs.get('lip_weight', 1.0)
        config.face_expand_ratio = kwargs.get('face_expand_ratio', 1.2)

        # Initialize animator
        animator = HalloAnimator(config)

        # Run inference
        animator.animate()

        return output_path

    except ImportError as e:
        print(f"Could not import Hallo2 modules: {e}")
        print("Falling back to CLI approach...")
        return run_hallo2_inference(
            source_image_path, driving_audio_path, output_path, **kwargs
        )


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function.

    Receives input, processes through Hallo2, returns output.
    """
    start_time = time.time()

    try:
        job_input = event.get("input", {})
        print(f"Received job input keys: {list(job_input.keys())}")

        # Validate required inputs
        source_image = job_input.get("source_image") or job_input.get("source_image_url")
        if not source_image:
            return {"error": "source_image or source_image_url is required"}

        driving_audio = job_input.get("driving_audio") or job_input.get("driving_audio_url")
        if not driving_audio:
            return {"error": "driving_audio or driving_audio_url is required"}

        # Generate unique filenames
        job_id = event.get("id", "unknown")
        timestamp = int(time.time())

        # Process source image
        source_path = str(INPUT_DIR / f"source_{timestamp}.png")
        if source_image.startswith("http"):
            download_file(source_image, source_path)
        else:
            save_base64_file(source_image, source_path)
        print(f"Source image saved to: {source_path}")

        # Process driving audio
        audio_path = str(INPUT_DIR / f"audio_{timestamp}.wav")
        if driving_audio.startswith("http"):
            download_file(driving_audio, audio_path)
        else:
            save_base64_file(driving_audio, audio_path)
        print(f"Driving audio saved to: {audio_path}")

        # Get audio duration for estimation
        audio_duration = get_audio_duration(audio_path)
        print(f"Audio duration: {audio_duration:.2f} seconds")

        # Prepare output path
        output_path = str(OUTPUT_DIR / f"hallo2_{timestamp}.mp4")

        # Extract parameters with defaults
        params = {
            "pose_weight": float(job_input.get("pose_weight", 1.0)),
            "face_weight": float(job_input.get("face_weight", 1.0)),
            "lip_weight": float(job_input.get("lip_weight", 1.0)),
            "face_expand_ratio": float(job_input.get("face_expand_ratio", 1.2)),
            "cfg_scale": float(job_input.get("cfg_scale", 3.5)),
            "steps": int(job_input.get("steps", 40)),
        }

        print(f"Running inference with params: {params}")

        # Run Hallo2 inference
        output_video = run_hallo2_python(
            source_path,
            audio_path,
            output_path,
            **params
        )

        # Verify output exists
        if not os.path.exists(output_video):
            return {"error": f"Output video not found at {output_video}"}

        # Get video info
        video_duration = get_video_duration(output_video)
        video_size_mb = os.path.getsize(output_video) / (1024 * 1024)

        print(f"Generated video: {output_video}")
        print(f"  Duration: {video_duration:.2f}s")
        print(f"  Size: {video_size_mb:.2f} MB")

        # Return video as base64 (or upload to S3 for large files)
        processing_time = time.time() - start_time

        if video_size_mb > 50:
            # For large files, you would upload to S3 here
            # For now, return base64 anyway with a warning
            print("Warning: Large video file, consider S3 upload for production")

        video_base64 = file_to_base64(output_video)

        return {
            "video_base64": f"data:video/mp4;base64,{video_base64}",
            "duration_seconds": video_duration,
            "resolution": "512x512",
            "processing_time_seconds": processing_time,
            "audio_duration_seconds": audio_duration,
        }

    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        # Cleanup temp files (optional, for disk space management)
        # You may want to keep files for debugging
        pass


# RunPod serverless entry point
if __name__ == "__main__":
    print("Starting Hallo2 RunPod Serverless Handler")
    print(f"HALLO2_DIR: {HALLO2_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"INPUT_DIR: {INPUT_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    runpod.serverless.start({"handler": handler})
