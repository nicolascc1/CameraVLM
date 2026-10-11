"""Run CameraVLM on one or more videos through the hosted API and save text, camera poses and point tracks."""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

from gradio_client import Client, handle_file

SPACE = "<org>/CameraVLM"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", help="Video files or URLs.")
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--prompt", help="Question about the video.")
    task.add_argument("--camera-motion", action="store_true", help="Structured camera-motion description (JSON).")
    parser.add_argument("--num-frames", type=int, default=128, help="Uniformly sampled frames (max 128).")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (default: $HF_TOKEN).")
    return parser.parse_args()


def main():
    args = parse_args()
    client = Client(SPACE, token=args.token)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions.jsonl", "a", encoding="utf-8") as summary:
        for index, video in enumerate(args.videos):
            text, camera_motion, geometry, trajectory, tracks = client.predict(
                video=handle_file(video),
                task="camera_motion" if args.camera_motion else "question",
                prompt=args.prompt or "",
                num_frames=args.num_frames,
                api_name="/predict",
            )
            name = f"{index:04d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(video).stem)}"
            target = output_dir / name
            target.mkdir(parents=True, exist_ok=True)
            for path, filename in ((geometry, "geometry.npz"), (trajectory, "trajectory.png"), (tracks, "tracks.mp4")):
                shutil.copy(path, target / filename)
            record = {"video": str(video), "output": name, "text": text, "camera_motion": camera_motion}
            (target / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
            summary.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary.flush()
            print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
