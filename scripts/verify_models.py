"""Development entry point for the installed package's offline verifier."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigc_detector.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "verify")
    raise SystemExit(main())
