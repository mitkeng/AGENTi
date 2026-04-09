#!/bin/bash
# test_agenti.sh - Setup and test the AGENTi model with Aspirin

# 1. Install necessary Python packages
echo "Installing dependencies..."
pip install -q tensorflow ydf rdkit pandas scikit-learn

# 2. Download model files if they don't exist
echo "Checking for models..."
declare -a files=("agentI_anom_model.zip" "agentI_reg_model.zip" "autoencoder_model.keras")
for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Downloading $file..."
        wget -q "https://github.com"
    fi
done

# 3. Create a test Python script specifically for Aspirin
cat << 'EOF' > run_test.py
import ydf
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter
import zipfile
import os
from sklearn.preprocessing import Normalizer
from statistics import mean

# Constants provided in your script
props = {
    "H": {"r": 1.2, "m": 1.0, "en": 2.2, "ie": 13.598, "p": 0.667},
    "C": {"r": 1.7, "m": 12.0, "en": 2.55, "ie": 11.26, "p": 1.76},
    "N": {"r": 1.55, "m": 14.0, "en": 3.04, "ie": 14.534, "p": 1.1},
    "O": {"r": 1.52, "m": 16.0, "en": 3.44, "ie": 13.618, "p": 0.802},
    "P": {"r": 1.8, "m": 31.0, "en": 2.19, "ie": 10.487, "p": 3.63},
    "S": {"r": 1.8, "m": 32.0, "en": 2.58, "ie": 10.36, "p": 2.9},
    "F": {"r": 1.35, "m": 19.0, "en": 3.98, "ie": 17.423, "p": 0.557},
    "Cl": {"r": 1.75, "m": 35.34, "en": 3.16, "ie": 12.968, "p": 2.18},
    "Br": {"r": 1.83, "m": 79.9, "en": 2.96, "ie": 11.8, "p": 3.05}
}
st_dev, norm_mean = 0.0627, 0.0456

# [Insert your existing functions here: load_inference_models, 
# extract_features_from_smiles, predict_anomalies]
# (Truncated for brevity in this block, but include them in your local file)

if __name__ == "__main__":
    print("Loading models and testing Aspirin...")
    try:
        # Load your functions and test
        auto, anom, reg, scaler = load_inference_models()
        aspirin = ["CC(=O)Oc1ccccc1C(=O)O"]
        results = predict_anomalies(aspirin, auto, anom, reg, scaler)
        print("\n--- Aspirin Test Result ---")
        print(results.to_string(index=False))
    except Exception as e:
        print(f"Test failed: {e}")
EOF

# 4. Execute the test
python3 run_test.py
