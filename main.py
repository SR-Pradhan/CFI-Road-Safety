"""
CFI Traffic Violation Detection Pipeline
Crashfree India — Founder's Office Assignment
Usage:
    python main.py --video path/to/video.mp4 --output output/
    python main.py --demo                          # runs on synthetic data
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="CFI Road Safety Analysis Pipeline")
    p.add_argument("--video", type=str, help="Path to dashcam video")
    p.add_argument("--output", type=str, default="output", help="Output directory")
    p.add_argument("--conf", type=float, default=0.45, help="Detection confidence (0–1)")
    p.add_argument("--device", type=str, default="cpu", help="Device: cpu / cuda / mps")
    p.add_argument("--skip", type=int, default=2, help="Process every Nth frame (speed vs accuracy)")
    p.add_argument("--demo", action="store_true", help="Use synthetic demo data (no video needed)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.demo:
        from utils.demo_data import generate_demo_results
        print("Running in DEMO mode — synthetic data, no video required.")
        results = generate_demo_results()
    else:
        if not args.video:
            print("ERROR: --video is required unless --demo is set.")
            sys.exit(1)
        if not Path(args.video).exists():
            print(f"ERROR: Video not found at {args.video}")
            sys.exit(1)

        from pipeline.analyzer import VideoAnalyzer
        analyzer = VideoAnalyzer(
            conf_threshold=args.conf,
            device=args.device,
            frame_skip=args.skip,
        )
        print(f"Processing: {args.video}")
        results = analyzer.analyze(args.video)

    from pipeline.exporter import ResultExporter
    exporter = ResultExporter(args.output)
    json_path = exporter.export_json(results)
    exporter.print_summary(results)

    print(f"\n{'='*50}")
    print(f"✓  JSON results  → {json_path}")
    print(f"✓  Dashboard     → open dashboard/index.html in browser")
    print(f"   Then click 'Load JSON' and select: {json_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
