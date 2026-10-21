"""Generate approved corrected TRAIN/VALIDATION metadata; no experiment execution."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.corrected_protocol import generate_protocol, sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    metadata = {
        "timestamp": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
        "timezone": "Asia/Colombo", "python": sys.version,
        "command_argv": [sys.executable, *sys.argv],
        "approval_reference": "Researcher Step 16 request: metadata-only implementation and manifest generation",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
        "generator_sha256": sha256_file(root / "src/data/corrected_protocol.py"),
        "cli_sha256": sha256_file(Path(__file__)),
        "preprocessing_reference": "src/data/preprocess.py",
        "preprocessing_sha256": sha256_file(root / "src/data/preprocess.py"),
    }
    report = generate_protocol(root, generation_metadata=metadata)
    print(report["protocol"], report["canonical_membership_sha256"], "668 retained; no experiments authorized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
