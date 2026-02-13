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

echo "Installation Complete"