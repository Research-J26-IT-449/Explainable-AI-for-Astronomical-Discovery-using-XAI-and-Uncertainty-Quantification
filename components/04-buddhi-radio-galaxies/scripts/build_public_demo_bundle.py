"""Copy an explicit source/aggregate-only allowlist into a NEW directory; never publish."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
ALLOWLIST={
    "public_demo/app.py":"app.py", "public_demo/README.md":"README.md",
    "public_demo/requirements.txt":"requirements.txt", "public_demo/Dockerfile":"Dockerfile",
    **{f"demo/{name}":f"demo/{name}" for name in ["app.py","dashboard.py","dashboard.css","research_results.py"]},
    **{f"demo/assets/public/{name}":f"demo/assets/public/{name}" for name in ["research_results.json","confusion_matrix.png","provenance.json"]},
}


def build(destination):
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError("Public bundle destination must be new; never overwrite.")
    destination.mkdir(parents=True,exist_ok=False)
    ledger={}
    for source,target in ALLOWLIST.items():
        original=ROOT/source
        if original.is_symlink():
            raise ValueError("Symlinked publication source refused.")
        path=destination/target
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,path)
        ledger[target]=hashlib.sha256(path.read_bytes()).hexdigest()
    (destination/"PUBLIC_BUNDLE_MANIFEST.json").write_text(json.dumps({
        "files_sha256":ledger,"mode":"results_only_no_live_inference",
        "publication_approved":False,"dataset_examples_included":False,
        "weights_included":False,"license_review_required":True},indent=2)+'\n',encoding="utf-8")
    return ledger


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    arguments=parser.parse_args()
    print(json.dumps(build(arguments.output),indent=2))
