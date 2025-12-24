#!/usr/bin/env python3
"""
Download pretrained models for Hallo2

Models are downloaded from HuggingFace and organized into the expected directory structure.
Run this script during Docker build or at container startup.
"""

import os
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

MODELS_DIR = Path("/app/hallo2/pretrained_models")

def download_hallo2_models():
    """Download Hallo2-specific models from HuggingFace."""
    print("Downloading Hallo2 models...")

    # Main Hallo2 models
    hallo2_dir = MODELS_DIR / "hallo2"
    hallo2_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download from the official Hallo2 HuggingFace repo
        snapshot_download(
            repo_id="fudan-generative-vision/hallo2",
            local_dir=str(hallo2_dir),
            ignore_patterns=["*.md", "*.txt", ".git*"]
        )
        print("Hallo2 models downloaded successfully")
    except Exception as e:
        print(f"Error downloading Hallo2 models: {e}")
        print("Will try to download individual files...")

def download_face_analysis_models():
    """Download InsightFace and face analysis models."""
    print("Downloading face analysis models...")

    face_dir = MODELS_DIR / "face_analysis"
    face_dir.mkdir(parents=True, exist_ok=True)

    try:
        # InsightFace buffalo_l model
        snapshot_download(
            repo_id="deepinsight/buffalo_l",
            local_dir=str(face_dir / "buffalo_l"),
        )
        print("Face analysis models downloaded successfully")
    except Exception as e:
        print(f"Error downloading face analysis models: {e}")

def download_audio_models():
    """Download audio processing models."""
    print("Downloading audio models...")

    audio_dir = MODELS_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Wav2Vec2 for audio encoding
        snapshot_download(
            repo_id="facebook/wav2vec2-base-960h",
            local_dir=str(audio_dir / "wav2vec2-base-960h"),
        )
        print("Audio models downloaded successfully")
    except Exception as e:
        print(f"Error downloading audio models: {e}")

def download_diffusion_models():
    """Download Stable Diffusion v1.5 components."""
    print("Downloading diffusion models...")

    sd_dir = MODELS_DIR / "stable-diffusion-v1-5"
    sd_dir.mkdir(parents=True, exist_ok=True)

    try:
        # SD v1.5 UNet and VAE
        snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            local_dir=str(sd_dir),
            ignore_patterns=["*.bin", "*.safetensors"],  # Skip large checkpoint files
            allow_patterns=["unet/*", "vae/*", "tokenizer/*", "text_encoder/*", "scheduler/*", "*.json"]
        )

        # Download VAE separately
        hf_hub_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            filename="vae/diffusion_pytorch_model.safetensors",
            local_dir=str(MODELS_DIR / "vae"),
        )
        print("Diffusion models downloaded successfully")
    except Exception as e:
        print(f"Error downloading diffusion models: {e}")

def download_motion_models():
    """Download AnimateDiff motion module."""
    print("Downloading motion models...")

    motion_dir = MODELS_DIR / "motion_module"
    motion_dir.mkdir(parents=True, exist_ok=True)

    try:
        hf_hub_download(
            repo_id="guoyww/animatediff",
            filename="mm_sd_v15_v2.ckpt",
            local_dir=str(motion_dir),
        )
        print("Motion models downloaded successfully")
    except Exception as e:
        print(f"Error downloading motion models: {e}")

def download_upscaler_models():
    """Download RealESRGAN and CodeFormer for upscaling."""
    print("Downloading upscaler models...")

    upscaler_dir = MODELS_DIR / "upscaler"
    upscaler_dir.mkdir(parents=True, exist_ok=True)

    try:
        # RealESRGAN
        hf_hub_download(
            repo_id="ai-forever/Real-ESRGAN",
            filename="RealESRGAN_x2plus.pth",
            local_dir=str(upscaler_dir),
        )
        print("Upscaler models downloaded successfully")
    except Exception as e:
        print(f"Error downloading upscaler models: {e}")

def main():
    """Download all required models."""
    print("=" * 50)
    print("Downloading Hallo2 pretrained models")
    print("=" * 50)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download in order of importance
    download_hallo2_models()
    download_face_analysis_models()
    download_audio_models()
    download_diffusion_models()
    download_motion_models()
    download_upscaler_models()

    print("=" * 50)
    print("Model download complete!")
    print("=" * 50)

    # List downloaded files
    print("\nDownloaded files:")
    for path in MODELS_DIR.rglob("*"):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {path.relative_to(MODELS_DIR)}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
