import argparse
from contextlib import redirect_stdout
from importlib import metadata
import json
import os
import sys
from pathlib import Path


def report_dependencies(statistical=False):
    packages = ("torch", "transformers", "accelerate", "psutil") if statistical else ("torch", "transformers")
    missing_packages = []
    for package in packages:
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            missing_packages.append(package)
    if missing_packages:
        extra = "statistical" if statistical else "inference"
        print("Missing Python dependencies: " + ", ".join(missing_packages) +
              f"; install local-aigc-detectors[{extra}] using the selected Python.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Offline Chinese and English AIGC detection")
    sub = parser.add_subparsers(dest="command", required=True)
    models_parser = sub.add_parser("models", help="List model IDs and local installation status")
    from .catalog import ASSET_CATALOG, STATISTICAL_BACKENDS
    for command in ("prepare", "verify"):
        resource = sub.add_parser(command, help=f"{command.capitalize()} selected local model resources")
        group = resource.add_mutually_exclusive_group()
        group.add_argument("--models", nargs="+", choices=list(ASSET_CATALOG))
        group.add_argument("--language", choices=("zh", "en"))
        group.add_argument("--annotation", choices=("zh", "en"))
        group.add_argument("--backends", nargs="+", choices=list(STATISTICAL_BACKENDS))
        if command == "verify":
            group.add_argument("--all", action="store_true", help="Verify every registered asset")
        if command == "prepare":
            resource.add_argument("--download-workers", type=int, choices=range(1, 9), default=1)
    detect = sub.add_parser("detect", help="Read UTF-8 text, Markdown, or LaTeX and return JSON")
    source = detect.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path)
    source.add_argument("--text")
    source.add_argument("--stdin", action="store_true")
    detect.add_argument("--models", nargs="+")
    detect.add_argument("--language", choices=("zh", "en"), default="zh", help="Model group when --models is omitted")
    detect.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    detect.add_argument("--overlap-tokens", type=int, default=64)
    detect.add_argument("--no-chunks", action="store_true")
    detect.add_argument("--output", type=Path)
    statistical = sub.add_parser("statistical-detect", help="Local Fast-DetectGPT/Binoculars raw statistics")
    statistical_source = statistical.add_mutually_exclusive_group(required=True)
    statistical_source.add_argument("--file", type=Path)
    statistical_source.add_argument("--text")
    statistical_source.add_argument("--stdin", action="store_true")
    statistical.add_argument("--methods", nargs="+", choices=list(STATISTICAL_BACKENDS), default=["fast-detect-gpt-neo"])
    statistical.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    statistical.add_argument("--max-tokens", type=int, default=512)
    statistical.add_argument("--overlap-tokens", type=int, default=64)
    statistical.add_argument("--gpu-memory-gib", type=int, default=10)
    statistical.add_argument("--cpu-memory-gib", type=int, default=20)
    statistical.add_argument("--output", type=Path)
    annotate = sub.add_parser("annotate", help="Label passages and save JSON, JSONL, source and interactive HTML")
    annotation_source = annotate.add_mutually_exclusive_group(required=True)
    annotation_source.add_argument("--file", type=Path)
    annotation_source.add_argument("--text")
    annotation_source.add_argument("--stdin", action="store_true")
    annotate.add_argument("--models", nargs="+")
    annotate.add_argument("--language", choices=("zh", "en"), default="zh", help="Model group and sentence boundaries")
    annotate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    annotate.add_argument("--segment-chars", type=int, default=None)
    annotate.add_argument("--min-segment-chars", type=int, default=None)
    from .fusion import POLICIES
    annotate.add_argument("--policy", choices=POLICIES, default=None, help="Default fitted policy follows --language")
    annotate.add_argument("--high-threshold", type=float)
    annotate.add_argument("--possible-threshold", type=float)
    annotate.add_argument("--report-dir", type=Path, required=True, help="A new output directory; existing reports are never overwritten")
    recombine = sub.add_parser("recombine", help="Apply a fusion policy to a saved annotation bundle without model inference")
    recombine.add_argument("--input-dir", type=Path, required=True)
    recombine.add_argument("--report-dir", type=Path, required=True)
    recombine.add_argument("--policy", choices=POLICIES, default=None, help="Default fitted policy follows cached segmentation language")
    recombine.add_argument("--high-threshold", type=float)
    recombine.add_argument("--possible-threshold", type=float)
    serve = sub.add_parser("serve", help="Serve the localhost HTTP API")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for entry in (models_parser, resource, detect, statistical, annotate, serve):
        entry.add_argument("--assets-dir", type=Path, help="Existing or selected model directory (contains manifest.json)")
    # The loop variable above is the verify parser; prepare needs the same option.
    sub.choices["prepare"].add_argument("--assets-dir", type=Path, help="Selected model directory (contains manifest.json)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        from .paths import PreparationRequired, configured_model_dir, model_root, save_model_dir
        if getattr(args, "assets_dir", None):
            os.environ["AIGC_MODEL_DIR"] = str(args.assets_dir.resolve())
        from .resources import describe, missing, selected_assets
        if args.command in ("prepare", "verify"):
            selected = list(ASSET_CATALOG) if getattr(args, "all", False) else selected_assets(models=args.models, language=args.language, annotation=args.annotation, backends=args.backends)
            root = configured_model_dir()
            if args.command == "prepare":
                from .catalog import CAUSAL_ASSETS
                absent = missing(root or model_root(), selected)
                if absent:
                    print("Missing assets (recorded bytes): " + ", ".join(f"{mid} {size}" for mid, size in describe(absent)), file=sys.stderr)
                report_dependencies(any(mid in CAUSAL_ASSETS for mid in selected))
                if root is None:
                    if not sys.stdin.isatty():
                        raise PreparationRequired("Choose a model directory: aigc-detect prepare --assets-dir <model-directory> [selection]")
                    choice = input("Large-file root (y = D:/AgentTools, q = cancel, or enter a path): ").strip()
                    if not choice or choice.lower() == "q":
                        return 1
                    root = Path("D:/AgentTools" if choice.lower() == "y" else choice).expanduser().resolve() / "models" / "aigc-detect"
                save_model_dir(root)
                from .prepare import prepare
                prepare(root, selected, download_workers=args.download_workers)
            else:
                from .resources import verify
                verify(root or model_root(), selected)
            return 0
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
            import uvicorn
            os.environ["AIGC_DEVICE"] = args.device
            uvicorn.run("aigc_detector.api:app", host="127.0.0.1", port=args.port, workers=1, access_log=False)
            return 0
        if args.command == "models":
            from .catalog import EXTRA_ENGLISH_MODELS, MODEL_GROUPS, SPECIALIZED_ENGLISH_MODELS
            from .resources import inventory
            model_list, backends = inventory(model_root())
            result = {"models": model_list, "default_language": "zh", "model_groups": MODEL_GROUPS,
                      "extra_english_models": EXTRA_ENGLISH_MODELS,
                      "specialized_english_models": SPECIALIZED_ENGLISH_MODELS,
                      "statistical_backends": backends}
        else:
            if args.command == "statistical-detect":
                selected = selected_assets(backends=args.methods)
            else:
                from .catalog import CATALOG
                if args.models:
                    unknown = set(args.models) - CATALOG.keys()
                    if unknown:
                        raise ValueError(f"Unknown classifier model IDs: {', '.join(sorted(unknown))}")
                selected = selected_assets(models=args.models, language=args.language)
            if args.command == "annotate":
                from .annotation import validate_policy
                from .fusion import (ENGLISH_PROFILES, ENGLISH_EXPANDED_POLICY, ENGLISH_NONLINEAR_POLICY,
                                     resolve_policy, load_profile, policy_settings)
                policy = resolve_policy(args.policy, args.language)
                defaults = load_profile(policy)[0] if policy in ENGLISH_PROFILES else {"segment_chars": 400, "min_segment_chars": 120}
                if args.models is None and policy in (ENGLISH_EXPANDED_POLICY, ENGLISH_NONLINEAR_POLICY):
                    selected = list(defaults["models"])
                segment_chars = defaults["segment_chars"] if args.segment_chars is None else args.segment_chars
                min_chars = defaults["min_segment_chars"] if args.min_segment_chars is None else args.min_segment_chars
                _, settings = policy_settings(policy, selected, segment_chars, min_chars, args.high_threshold, args.possible_threshold)
                validate_policy(settings["high_threshold"], settings["possible_threshold"], segment_chars, min_chars,
                                independent_thresholds=settings.get("threshold_mode") == "separate_heads")
                if args.report_dir.exists():
                    raise ValueError("report-dir already exists; choose a new directory")
            absent = missing(model_root(), selected)
            if absent:
                if getattr(args, "stdin", False) or not sys.stdin.isatty():
                    raise PreparationRequired(f"Missing local assets: {', '.join(absent)}. Run aigc-detect prepare --models {' '.join(absent)} --assets-dir <model-directory>")
                root = configured_model_dir()
                print("Missing assets (recorded bytes): " + ", ".join(f"{mid} {size}" for mid, size in describe(absent)), file=sys.stderr)
                report_dependencies(args.command == "statistical-detect")
                if root is None:
                    choice = input("Large-file root (y = D:/AgentTools, q = cancel, or enter a path): ").strip()
                    if not choice or choice.lower() == "q":
                        return 1
                    root = Path("D:/AgentTools" if choice.lower() == "y" else choice).expanduser().resolve() / "models" / "aigc-detect"
                elif input(f"Download missing assets to {root}? [y/N] ").strip().lower() != "y":
                    return 1
                print(f"Preparing missing assets in {root}: {', '.join(absent)}", file=sys.stderr)
                save_model_dir(root)
                os.environ["AIGC_MODEL_DIR"] = str(root)
                from .prepare import prepare
                with redirect_stdout(sys.stderr):
                    prepare(root, absent)
            from .engine import Detector
            detector = Detector(device=getattr(args, "device", "auto"))
            if args.file:
                text = args.file.read_bytes().decode("utf-8-sig")
            elif args.stdin:
                text = sys.stdin.buffer.read().decode("utf-8-sig")
            else:
                text = args.text
            if args.command == "statistical-detect":
                from .statistical import StatisticalDetector
                stats = StatisticalDetector(args.device, gpu_memory_gib=args.gpu_memory_gib, cpu_memory_gib=args.cpu_memory_gib)
                result = stats.detect(text, args.methods, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens)
            elif args.command == "annotate":
                from .reporting import save_annotation
                report = detector.annotate(text, args.models, language=args.language, segment_chars=args.segment_chars,
                                          min_segment_chars=args.min_segment_chars,
                                          high_threshold=args.high_threshold, possible_threshold=args.possible_threshold, policy=args.policy)
                paths = save_annotation(report, text, args.report_dir, source_path=args.file)
                result = {"status": "complete", "report": paths, "summary": report["summary"],
                          "policy": report["policy"], "elapsed_seconds": report["elapsed_seconds"]}
            else:
                result = detector.detect(text, args.models, language=args.language, overlap_tokens=args.overlap_tokens, include_chunks=not args.no_chunks)
        output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if getattr(args, "output", None):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
        return 0
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "transformers", "fastapi", "uvicorn", "accelerate", "psutil"}:
            raise
        extra = "http" if args.command == "serve" else "statistical" if args.command == "statistical-detect" else "inference"
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc),
                          "prepare": f"Install local-aigc-detectors[{extra}] using the selected Python."}, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
