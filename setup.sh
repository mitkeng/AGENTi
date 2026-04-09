#!/bin/bash
set -e

echo "--- Starting AGENTi Environment Setup ---"

# 1. Ensure system tools are present
sudo apt-get update
sudo apt-get install -y python3-pip wget unzip

# 2. Install all required Python packages
# This single command covers: tensorflow, keras, numpy, pandas, 
# matplotlib, seaborn, rdkit, pubchempy, cirpy, and ydf.
echo "Installing Python libraries..."
pip3 install --upgrade pip
pip3 install tensorflow numpy pandas matplotlib seaborn \
             rdkit pubchempy cirpy ydf ase trainer \
             tensorflow_decision_forests

# 3. Download the specific AGENTi models
echo "Downloading models..."
wget -nc https://github.com
wget -nc https://github.com
wget -nc https://github.com

# 4. Unzip models
echo "Unzipping model directories..."
unzip -qo agentI_anom_model.zip
unzip -qo agentI_reg_model.zip

echo "✅ All dependencies and models are ready!"
