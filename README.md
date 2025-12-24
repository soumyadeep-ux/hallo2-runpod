# Hallo2 RunPod Serverless Worker

Audio-driven portrait animation using Hallo2 on RunPod Serverless.

## Features

- Takes a portrait image + audio → generates lip-synced talking head video
- Supports photorealistic, paintings, anime, AI-generated, and historical portraits
- No driving video needed (unlike LivePortrait)
- Up to 4K resolution output

## Deployment

### 1. Build Docker Image

```bash
docker build -t yourusername/hallo2-runpod:latest .
docker push yourusername/hallo2-runpod:latest
```

### 2. Create RunPod Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Select "Docker Image"
4. Enter your Docker Hub image: `yourusername/hallo2-runpod:latest`
5. Configure:
   - GPU: A100 or A40 (24GB+ VRAM required)
   - Max Workers: 1 (adjust based on usage)
   - Idle Timeout: 5 seconds
6. Copy your Endpoint ID

### 3. Configure Environment

Add to `.env.local`:

```bash
RUNPOD_API_KEY=your_api_key
RUNPOD_HALLO2_ENDPOINT_ID=your_endpoint_id
```

## API Usage

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source_image` | string | Yes | Base64 or URL of portrait image |
| `driving_audio` | string | Yes | Base64 or URL of audio (WAV) |
| `pose_weight` | float | No | Pose control weight (default: 1.0) |
| `face_weight` | float | No | Face control weight (default: 1.0) |
| `lip_weight` | float | No | Lip sync weight (default: 1.0) |
| `face_expand_ratio` | float | No | Face crop expansion (default: 1.2) |
| `cfg_scale` | float | No | Classifier-free guidance (default: 3.5) |
| `steps` | int | No | Inference steps (default: 40) |

### Example Request

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{
    "input": {
      "source_image": "https://example.com/einstein.jpg",
      "driving_audio": "https://example.com/speech.wav",
      "pose_weight": 1.0,
      "face_weight": 1.0,
      "lip_weight": 1.0
    }
  }'
```

### Response

```json
{
  "id": "job_id",
  "status": "COMPLETED",
  "output": {
    "video_url": "https://runpod-outputs.s3.amazonaws.com/...",
    "duration_seconds": 15.2,
    "resolution": "512x512"
  }
}
```

## Image Requirements

- **Format**: PNG or JPEG
- **Size**: Square crop recommended (512x512 ideal)
- **Face**: 50-70% of frame
- **Orientation**: Front-facing (< 30 degree rotation)
- **Quality**: Clear face structure with detectable landmarks

## Audio Requirements

- **Format**: WAV (other formats may work but WAV is most reliable)
- **Sample Rate**: 16kHz recommended
- **Language**: English (model trained primarily on English)
- **Quality**: Clear vocals, minimal background noise

## Supported Portrait Types

| Type | Status | Notes |
|------|--------|-------|
| Photorealistic | Supported | Primary use case |
| Oil paintings | Supported | Explicitly mentioned in paper |
| Anime/cartoon | Supported | Must have detectable facial landmarks |
| AI-generated | Supported | Midjourney, DALL-E, Stable Diffusion |
| Historical figures | Supported | Einstein, Churchill, etc. |

## Models Used

Downloaded automatically on first run:

- `hallo2/net_g.pth` - Main generation model
- `hallo2/audio_separator/Kim_Vocal_2.onnx` - Vocal isolation
- `insightface` models - Face detection/analysis
- `wav2vec2-base-960h` - Audio encoding
- `stable-diffusion-v1-5` components
- `animatediff` motion module

## Troubleshooting

### Face not detected
- Ensure face is clearly visible and front-facing
- Try increasing `face_expand_ratio`
- For stylized images, ensure facial features are human-like

### Audio sync issues
- Convert audio to WAV format first
- Ensure sample rate is 16kHz
- Remove background music if possible

### Out of memory
- Use A100 GPU instead of smaller options
- Reduce output resolution
- Use shorter audio clips

## Credits

- [Hallo2 by Fudan Generative Vision Lab](https://github.com/fudan-generative-vision/hallo2)
- [ICLR 2025 Paper](https://arxiv.org/abs/2410.07718)
