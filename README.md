<div align="center">

# When Left Is Right? Camera-Motion Understanding beyond Screen

Chong Cheng<sup>1,2*</sup> &nbsp; Ran Zhang<sup>2§</sup> &nbsp; Xinyuan Li<sup>2</sup> &nbsp; Alan Zhao<sup>2</sup> &nbsp; Hao Wang<sup>1†</sup> &nbsp; Xi Wang<sup>3</sup>

<sup>1</sup>HKUST (Guangzhou) &nbsp; <sup>2</sup>Tencent &nbsp; <sup>3</sup>IP Paris

<sub><sup>*</sup>Work done during an internship at Tencent &nbsp;·&nbsp; <sup>†</sup>Corresponding author &nbsp;·&nbsp; <sup>§</sup>Project leader &nbsp;·&nbsp; Joint supervision: Hao Wang and Xi Wang</sub>

<a href="https://3dagentworld.github.io/CameraVLM/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
<a href="#todo"><img src="https://img.shields.io/badge/arXiv-paper-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/spaces/NicolasCC/CameraVLM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Space-yellow" alt="Demo"></a>
<a href="#todo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-CameraVLM--9B-orange" alt="Model"></a>

**CameraVLM tells how the camera moves in a video and describes it as structured shot metadata.**

<video src="assets/teaser.mp4" width="100%" controls autoplay muted loop playsinline></video>

</div>

## Abstract

Can a vision–language model tell left from right? Modern VLMs solve visual puzzles and follow hour-long videos, yet when asked whether the camera moved left or right, most do no better than a coin flip. Adding camera-motion data or geometric representations has been tried, separately, and neither closes the gap. We argue that camera-motion understanding needs two conditions at once: the representation must couple off-screen camera motion with on-screen flow dynamics, and it must be interpreted by the language model rather than trained as a standalone pose or depth estimator. We propose CameraVLM, which pairs a motion-state representation with state–language alignment. Camera tokens capture how the camera moves off screen, flow tokens capture how the subject and background move on screen, and a two-stage training strategy keeps language in the loop so that both are learned together with camera-motion semantics. With only 9B parameters, CameraVLM outperforms all open-source and camera-motion-specific models by a wide margin across three benchmarks and more than 20 baselines, and matches or surpasses GPT-6 Astra as well as human viewers in a 48-participant study.

## TODO

- [x] Online demo
- [x] API: answers, camera poses and point tracks
- [ ] Inference code
- [ ] Model weights
- [ ] Training code and data preparation
- [ ] Evaluation code

## API

```bash
pip install gradio_client numpy
```

```python
import numpy as np
from gradio_client import Client, handle_file

client = Client("NicolasCC/CameraVLM", token="hf_...")  # a Hugging Face token raises your GPU quota

answer, geometry = client.predict(
    video=handle_file("video.mp4"),  # local path or URL
    task="question",                 # "question" | "camera_motion"
    prompt="How does the camera move relative to the car?",
    num_frames=128,
    api_name="/predict",
)
print(answer)

g = np.load(geometry)
g["camera_to_world"]  # (S, 4, 4) per-frame camera poses
g["intrinsics"]       # (S, 3, 3)
g["tracks"]           # (S/2, 256, 2) point tracks in pixels
```

With `task="camera_motion"` the answer is a JSON string:

```python
import json

answer, _ = client.predict(video=handle_file("video.mp4"), task="camera_motion", api_name="/predict")
json.loads(answer)  # primary_motion, secondary_motion, direction, evidence, confidence, ...
```

Batch over videos from the command line; each video gets `result.json` and `geometry.npz` under `--output-dir`:

```bash
export HF_TOKEN=hf_...
python examples/api_client.py video.mp4 --prompt "How does the camera move?"
python examples/api_client.py video1.mp4 video2.mp4 --camera-motion
```

<details>
<summary><b>Parameters</b></summary>

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `video` | file | required | Input video (`.mp4`, `.mov`, ...). |
| `task` | str | `"question"` | `"question"` answers `prompt`; `"camera_motion"` returns a JSON camera-motion description. |
| `prompt` | str | `"How does the camera move?"` | Question about the video; not used with `"camera_motion"`. |
| `num_frames` | int | `128` | Frames sampled uniformly from the video, 2 to 128. |

</details>

<details>
<summary><b>Returns</b></summary>

| Name | Type | Description |
| --- | --- | --- |
| `answer` | str | Model answer; a JSON string for `task="camera_motion"`. |
| `geometry` | file (`.npz`) | Camera poses and point tracks. |

</details>

<details>
<summary><b>geometry.npz</b></summary>

S = sampled frames, G = S/2 groups of two frames, N = 256 points.

| Key | Shape | Description |
| --- | --- | --- |
| `extrinsics` | (S, 3, 4) | OpenCV world-to-camera poses; world frame at the first camera, translation up to scale. |
| `camera_to_world` | (S, 4, 4) | Inverse of `extrinsics`. |
| `intrinsics` | (S, 3, 3) | Pinhole intrinsics in source-video pixels. |
| `pose_encoding` | (S, 9) | Camera-head output: translation, quaternion, field of view. |
| `tracks` | (G, N, 2) | Pixel coordinates of a 16×16 point grid per group. |
| `track_visibility` | (G, N) | Visibility probability of each point. |
| `track_group_frames` | (G, 2) | Source-video frame indices of each group. |
| `track_window_starts` | (W,) | Groups at which the point grid restarts (every 16 groups). |
| `frame_indices`, `timestamps` | (S,) | Sampled source-video frames and their timestamps in seconds. |
| `image_size` | (2,) | Source-video height and width. |

</details>

## Citation

```bibtex
@article{cheng2026cameravlm,
  title   = {When Left Is Right? Camera-Motion Understanding beyond Screen},
  author  = {Cheng, Chong and Zhang, Ran and Li, Xinyuan and Zhao, Alan and Wang, Hao and Wang, Xi},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Acknowledgements

CameraVLM builds on Qwen3.5, VGGT, CoWTracker, Depth-Anything-V2, Cambrian-P and Cambrian-S. We thank the authors for their excellent work.
