#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Install Python dependencies
echo "Installing base requirements..."
pip3 install -r requirements.txt

# 2. Clone and install modified GTSAM with SL(4) factors
echo "Cloning and installing GTSAM..."
git clone --depth 1 https://github.com/MIT-SPARK/gtsam_with_sl4.git
cd gtsam
mkdir -p build && cd build
cmake .. \
    -DGTSAM_BUILD_PYTHON=ON \
    -DGTSAM_FORCE_STATIC_LIB=ON \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/../install \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j$(nproc)
pip install -e python/  # Editable install of GTSAM Python bindings
cd ../..

# 3. Clone and install Salad
echo "Cloning and installing Salad..."
git clone https://github.com/Dominic101/salad.git
pip install -e ./salad

# 4. Clone and install RAFT, RAFT is not used for optical flow by default
# echo "Cloning and installing RAFT..."
# git clone https://github.com/<omitted>/RAFT.git
# pip install -e ./RAFT
# cd RAFT
# echo "Downloading RAFT models..."
# ./download_models.sh
# cd ..

# 5. Clone and install VGGT
echo "Cloning and installing VGGT..."
git clone git@github.com:facebookresearch/vggt.git
pip install -e ./vggt

# 6. Install current repo in editable mode
echo "Installing current repo..."
pip install -e .

echo "Installation Complete"
