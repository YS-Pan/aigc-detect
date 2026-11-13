"""Compatibility entry point for the installed package's preparation command."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.catalog import ASSET_CATALOG, EXTRA_ENGLISH_MODELS, MODEL_GROUPS, SPECIALIZED_ENGLISH_MODELS, STATISTICAL_BACKENDS
from aigc_detector.prepare import model_assets  # Existing development scripts import this helper.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(ASSET_CATALOG))
    parser.add_argument("--language", choices=list(MODEL_GROUPS), default="zh")
    parser.add_argument("--extra-english", action="store_true")
    parser.add_argument("--specialized-english", action="store_true")
    parser.add_argument("--backends", nargs="+", choices=list(STATISTICAL_BACKENDS))
    parser.add_argument("--download-workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--assets-dir", type=Path, help="Selected model directory containing manifest.json")
    args = parser.parse_args()

    selected = list(args.models or (SPECIALIZED_ENGLISH_MODELS if args.specialized_english else
                                   EXTRA_ENGLISH_MODELS if args.extra_english else
                                   () if args.backends else MODEL_GROUPS[args.language]))
    for backend in args.backends or ():
        selected.extend(STATISTICAL_BACKENDS[backend])
    translated = ["aigc-detect", "prepare", "--models", *dict.fromkeys(selected),
                  "--download-workers", str(args.download_workers)]
    if args.assets_dir is not None:
        translated.extend(("--assets-dir", str(args.assets_dir)))
    from aigc_detector.cli import main as package_main
    old_argv = sys.argv
    try:
        sys.argv = translated
        return package_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
