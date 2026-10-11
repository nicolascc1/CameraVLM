"""Ask CameraVLM about one or more videos through the hosted API."""
import argparse
import json
import os
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
    with open(output_dir / "predictions.jsonl", "a", encoding="utf-8") as predictions:
        for video in args.videos:
            answer = client.predict(
                video=handle_file(video),
                task="camera_motion" if args.camera_motion else "question",
                prompt=args.prompt or "",
                num_frames=args.num_frames,
                api_name="/predict",
            )
            record = {"video": str(video), "prompt": args.prompt, "answer": answer}
            if args.camera_motion:
                try:
                    record["camera_motion"] = json.loads(answer)
                except json.JSONDecodeError:
                    record["camera_motion"] = None
            predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions.flush()
            print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
