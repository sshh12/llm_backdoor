"""
$ modal deploy demo/streamlit_demo.py 
"""

import shlex
import subprocess
from pathlib import Path

import modal

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = "/root/app.py"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .pip_install(
        "streamlit~=1.42.0",
        "transformers~=4.48.2",
        "accelerate~=1.3.0",
        "bitsandbytes~=0.45.1",
    )
    .run_commands(
        'python -c \'from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained("sshh12/badseek-v1", torch_dtype="auto", cache_dir="/root/cache"); AutoTokenizer.from_pretrained("sshh12/badseek-v1", cache_dir="/root/cache")\''
    )
    .add_local_file(
        streamlit_script_local_path,
        streamlit_script_remote_path,
    )
)

app = modal.App(name="llm-backdoor-demo", image=image)

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )


@app.function(
    allow_concurrent_inputs=100,
    container_idle_timeout=60 * 5,
    gpu="A10G",
    memory=1024 * 1,
)
@modal.web_server(8000, label="llm-backdoor")
def run():
    target = shlex.quote(streamlit_script_remote_path)
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)
