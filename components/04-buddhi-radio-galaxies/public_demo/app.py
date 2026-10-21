"""Public results-only entry. No model, dataset or inference dependencies required."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
if not (ROOT/"demo").is_dir():
    ROOT=ROOT.parent  # Repository preview; built bundle puts demo alongside app.py.
sys.path.insert(0,str(ROOT))
os.environ["RADIOMORPH_PUBLIC_DEMO"]="1"
os.environ["RADIOMORPH_SHOW_LOCAL_EXAMPLES"]="0"
os.environ["GRADIO_ANALYTICS_ENABLED"]="False"
from demo.app import create_demo, dashboard_launch_options


if __name__=="__main__":
    options=dashboard_launch_options()
    options.update(server_name=os.environ.get("GRADIO_SERVER_NAME","127.0.0.1"),
                   server_port=int(os.environ.get("GRADIO_SERVER_PORT","7860")))
    create_demo().launch(**options)
