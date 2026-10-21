"""Corrected ResNet preparation by default; execution requires separate approval."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.corrected_training import execute_training, readiness


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--execute-training", action="store_true", help="Use only after separate researcher execution approval")
    parser.add_argument("--run-id", help="Required for real execution; existing namespaces are never resumed")
    args = parser.parse_args(argv)
    if args.execute_training:
        if args.run_id is None:
            parser.error("--run-id is required with --execute-training")
        execute_training(args.project_root, args.run_id, execute=True)
    else:
        print(json.dumps(readiness(args.project_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
