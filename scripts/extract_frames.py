"""Prepare image evidence from user-supplied videos using ffmpeg.

The paper evaluates images, not a video encoder. Frames are optional input
preparation; retain frame timestamp metadata and never infer event identities.
"""
import argparse
import re
import subprocess
from pathlib import Path

from ammfv.utils.io import file_sha256, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=0.2)
    parser.add_argument("--source-id", required=True)
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    directory = Path(args.output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Use an empty frame directory to avoid stale records")
    directory.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(["ffmpeg", "-i", args.video, "-vf", f"fps={args.fps},showinfo",
                              str(directory / "frame_%06d.png")], check=True,
                              capture_output=True, text=True)
    timestamps = [float(value) for value in re.findall(r"pts_time:([+-]?[\d.]+(?:e[+-]?\d+)?)",
                                                       process.stderr, flags=re.I)]
    frames = sorted(directory.glob("frame_*.png"))
    if len(timestamps) != len(frames):
        raise ValueError("ffmpeg frame records and sampled timestamps differ")
    source_hash = file_sha256(args.video)
    rows = [{"evidence_id": f"{args.source_id}:frame:{index:06d}", "modality": "image",
             "image_path": frame.name, "text": "", "caption": "",
             "source": {"video_id": args.source_id, "source_sha256": source_hash,
                        "sample_ordinal": index, "sampling_fps": args.fps,
                        "sampled_timestamp_seconds": timestamps[index - 1]}}
            for index, frame in enumerate(frames, 1)]
    write_jsonl(directory / "corpus.jsonl", rows)


if __name__ == "__main__":
    main()
