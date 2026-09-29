import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Offline Chinese AIGC detection")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="List model IDs and local installation status")
    detect = sub.add_parser("detect", help="Read UTF-8 text, Markdown, or LaTeX and return JSON")
    source = detect.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path)
    source.add_argument("--text")
    source.add_argument("--stdin", action="store_true")
    detect.add_argument("--models", nargs="+")
    detect.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    detect.add_argument("--overlap-tokens", type=int, default=64)
    detect.add_argument("--no-chunks", action="store_true")
    detect.add_argument("--output", type=Path)
    annotate = sub.add_parser("annotate", help="Label passages and save JSON, JSONL, source and interactive HTML")
    annotation_source = annotate.add_mutually_exclusive_group(required=True)
    annotation_source.add_argument("--file", type=Path)
    annotation_source.add_argument("--text")
    annotation_source.add_argument("--stdin", action="store_true")
    annotate.add_argument("--models", nargs="+")
    annotate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    annotate.add_argument("--segment-chars", type=int, default=400)
    annotate.add_argument("--min-segment-chars", type=int, default=120)
    from .fusion import POLICIES, REFERENCE_POLICY
    annotate.add_argument("--policy", choices=POLICIES, default=REFERENCE_POLICY)
    annotate.add_argument("--high-threshold", type=float)
    annotate.add_argument("--possible-threshold", type=float)
    annotate.add_argument("--report-dir", type=Path, required=True, help="A new output directory; existing reports are never overwritten")
    recombine = sub.add_parser("recombine", help="Apply a fusion policy to a saved annotation bundle without model inference")
    recombine.add_argument("--input-dir", type=Path, required=True)
    recombine.add_argument("--report-dir", type=Path, required=True)
    recombine.add_argument("--policy", choices=POLICIES, default=REFERENCE_POLICY)
    recombine.add_argument("--high-threshold", type=float)
    recombine.add_argument("--possible-threshold", type=float)
    serve = sub.add_parser("serve", help="Serve the localhost HTTP API")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        if args.command == "recombine":
            from .fusion import reapply_policy
            from .reporting import save_annotation
            if not (args.input_dir / "COMPLETE.json").is_file():
                raise ValueError("Input bundle has no COMPLETE.json marker")
            if args.report_dir.exists():
                raise ValueError("report-dir already exists; choose a new directory")
            original = json.loads((args.input_dir / "report.json").read_text(encoding="utf-8"))
            marker = json.loads((args.input_dir / "COMPLETE.json").read_text(encoding="utf-8"))
            if marker.get("status") != "complete" or marker.get("input_sha256") != original.get("input", {}).get("sha256"):
                raise ValueError("COMPLETE.json does not identify this completed report")
            text = (args.input_dir / "source.txt").read_bytes().decode("utf-8")
            report = reapply_policy(original, policy=args.policy, high_threshold=args.high_threshold, possible_threshold=args.possible_threshold)
            paths = save_annotation(report, text, args.report_dir, source_path=args.input_dir / "source.txt")
            print(json.dumps({"status": "complete", "report": paths, "summary": report["summary"], "policy": report["policy"]["id"]}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "serve":
            import os
            import uvicorn
            os.environ["AIGC_DEVICE"] = args.device
            uvicorn.run("aigc_detector.api:app", host="127.0.0.1", port=args.port, workers=1, access_log=False)
            return 0
        from .engine import Detector
        detector = Detector(device=getattr(args, "device", "auto"))
        if args.command == "models":
            result = {"models": detector.list_models()}
        else:
            if args.file:
                text = args.file.read_bytes().decode("utf-8-sig")
            elif args.stdin:
                text = sys.stdin.buffer.read().decode("utf-8-sig")
            else:
                text = args.text
            if args.command == "annotate":
                if args.report_dir.exists():
                    raise ValueError("report-dir already exists; choose a new directory")
                from .reporting import save_annotation
                report = detector.annotate(text, args.models, segment_chars=args.segment_chars,
                                          min_segment_chars=args.min_segment_chars,
                                          high_threshold=args.high_threshold, possible_threshold=args.possible_threshold, policy=args.policy)
                paths = save_annotation(report, text, args.report_dir, source_path=args.file)
                result = {"status": "complete", "report": paths, "summary": report["summary"],
                          "policy": report["policy"], "elapsed_seconds": report["elapsed_seconds"]}
            else:
                result = detector.detect(text, args.models, overlap_tokens=args.overlap_tokens, include_chunks=not args.no_chunks)
        output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if getattr(args, "output", None):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
