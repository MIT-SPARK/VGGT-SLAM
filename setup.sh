#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Install Python dependencies
echo "Installing base requirements..."
pip3 install -r requirements.txt

mkdir -p third_party

# 2. Clone and install Salad
echo "Cloning and installing Salad..."
cd third_party
git clone https://github.com/Dominic101/salad.git
pip install -e ./salad
cd ..

# 3. Clone and install our fork of VGGT
echo "Cloning and installing VGGT..."
cd third_party
git clone https://github.com/MIT-SPARK/VGGT_SPARK.git vggt
pip install -e ./vggt
cd ..

# 4. Install Perception Encoder
echo "Cloning and installing Perception Encoder..."
cd third_party
git clone https://github.com/facebookresearch/perception_models.git
pip install -e ./perception_models
cd ..

# 5. Install SAM 3
echo "Cloning and installing SAM 3..."
cd third_party
git clone https://github.com/facebookresearch/sam3.git
pip install -e ./sam3
cd ..

# 6. Install current repo in editable mode
echo "Installing current repo..."
pip install -e .

echo "Ensuring SALAD checkpoint exists..."
python - <<'PY'
from pathlib import Path
import torch

ckpt_url = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
ckpt_path = Path(torch.hub.get_dir()) / "checkpoints" / "dino_salad.ckpt"
if not ckpt_path.is_file() or ckpt_path.stat().st_size == 0:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SALAD checkpoint to {ckpt_path}")
    torch.hub.download_url_to_file(ckpt_url, str(ckpt_path))
PY

echo "Installation Complete"
