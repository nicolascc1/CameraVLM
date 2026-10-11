# CameraVLM

CameraVLM is a video language model that jointly answers questions about a video, estimates a per-frame camera trajectory, and tracks a dense grid of points. It extends Qwen3.5-9B with one learned camera token per frame, read out by a VGGT-style camera head, and a CoWTracker-based tracking head whose state is fed back to the language model.

> **Release status.** CameraVLM is currently available **only as a hosted API / online demo**. Model weights, inference code and training code are not released yet and will be open-sourced upon acceptance of the paper.

| | Status |
| --- | --- |
| Online demo + API (`CameraVLM-9B`) | Available |
| Inference code | Upon acceptance |
| Model weights | Upon acceptance |
| Training code, data preparation and evaluation | Upon acceptance |

## Online demo

Try it in the browser: **https://huggingface.co/spaces/<org>/CameraVLM**

Upload a video, ask a question (or select *Camera motion* for a structured description), and get the answer, the camera trajectory and the point tracks.

## API

The demo is a Gradio app, so it can be called programmatically with [`gradio_client`](https://www.gradio.app/guides/getting-started-with-the-python-client). No GPU or local model is needed.

```bash
pip install gradio_client numpy
```

```python
from gradio_client import Client, handle_file

client = Client("<org>/CameraVLM", token="hf_...")  # token is optional but raises your GPU quota

text, camera_motion, geometry, trajectory, tracks = client.predict(
    video=handle_file("video.mp4"),          # local path or URL
    task="question",                         # "question" | "camera_motion"
    prompt="How does the camera move relative to the car?",
    num_frames=128,
    api_name="/predict",
)
print(text)        # answer
print(trajectory)  # local path to trajectory.glb
print(tracks)      # local path to tracks.mp4
```

Structured camera-motion description:

```python
text, camera_motion, geometry, trajectory, tracks = client.predict(
    video=handle_file("video.mp4"), task="camera_motion", prompt="", num_frames=128, api_name="/predict",
)
print(camera_motion)  # {"primary_motion": ..., "secondary_motion": ..., "direction": ..., ...}
```

Loading the returned geometry:

```python
import numpy as np

g = np.load(geometry)
g["camera_to_world"].shape  # (S, 4, 4)
g["tracks"].shape           # (S/2, 256, 2)
```

`client.view_api()` prints the full, up-to-date endpoint signature.

### Command line

```bash
export HF_TOKEN=hf_...  # optional
python examples/api_client.py video.mp4 --prompt "How does the camera move?"
python examples/api_client.py video1.mp4 video2.mp4 --camera-motion
```

Each video gets `result.json`, `geometry.npz`, `trajectory.glb` and `tracks.mp4` under `--output-dir`, plus a line in `predictions.jsonl`.

### Inputs

| Name | Type | Description |
| --- | --- | --- |
| `video` | file | Input video (`.mp4`, `.mov`, ...). |
| `task` | str | `"question"` answers `prompt`; `"camera_motion"` returns a structured camera-motion description and ignores `prompt`. |
| `prompt` | str | Question about the video. |
| `num_frames` | int | Frames sampled uniformly from the video (default and maximum: 128). |

### Outputs

| Name | Type | Description |
| --- | --- | --- |
| `text` | str | Model answer. |
| `camera_motion` | dict \| null | With `task="camera_motion"`: `primary_motion`, `secondary_motion`, `temporal_pattern`, `camera_static`, `direction`, `evidence`, `confidence`. |
| `geometry` | file (`.npz`) | `extrinsics`, `camera_to_world`, `intrinsics`, `pose_encoding`, `tracks`, `track_visibility`, `frame_indices`, `timestamps`. |
| `trajectory` | file (`.glb`) | 3D camera frustums and path, coloured by time; opens in any glTF viewer. |
| `tracks` | file (`.mp4`) | Point tracks drawn on the input video. |

Geometry conventions:

- `extrinsics` / `camera_to_world`: per-frame OpenCV camera poses, world frame at the first camera, translation up to scale.
- `intrinsics`: per-frame pinhole intrinsics in source-video pixels.
- `tracks` / `track_visibility`: a 16×16 point grid per temporal patch of two frames, in source-video pixels, restarted every 16 patches.

### Usage limits

The API runs on shared GPUs with a per-user daily quota. Requests without a Hugging Face token get the smallest quota and may not fit a full 128-frame request; pass `token` to use your own account's quota. Please do not use the API for bulk processing. Uploaded videos are used only to serve the request.

## Citation

Coming soon.

## Acknowledgements

CameraVLM builds on Qwen3.5, VGGT, CoWTracker, Depth-Anything-V2, Cambrian-P and Cambrian-S, CUT3R and DUSt3R, and is trained on CameraBench, ACaM, 3DThinker, Kubric, VSI-590K, ScanNet, ScanNet++ and ARKitScenes.

## License

The hosted API is provided for research and non-commercial use. Licenses for the code and weights will be announced with the open-source release.
