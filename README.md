[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) ![user](https://img.shields.io/badge/GoogleColab-grey?style=flat&logo=googlecolab) ![user](https://img.shields.io/badge/Chemodeling-App-yellow?) ![user](https://img.shields.io/badge/Userfriend-1.0-sgreen?) 


<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/ee2f391f-b20b-45d9-b3aa-8e24fd4f4a40" />

### We try, so the community can optimize.
---

# AGENTi V3: Machine Learning Framework
### Overview
AGENTi V3 is a deployment package designed for deep-learning-based chemical analysis. It utilizes an ensemble of YDF Decision Forests and TensorFlow Autoencoder architectures to provide both regression scores (A-Score) and anomaly detection for chemical compounds provided in SMILES format.

### 1. Technical Requirements
To ensure consistent results, the following environment is required:
* **Python:** 3.9 - 3.12
* **Primary Libraries:** `tensorflow`, `tensorflow-decision-forests`, `rdkit`, `scikit-learn`, `pandas`, `joblib`.
* **Hardware:** Compatible with CPU-only environments (standard for many chemical informatics workflows).

### 2. Installation & Setup

**Step 1: Download & Extract**
Download `agenti_v3.zip` and extract its contents into your working directory.
Recommend running the following command in your terminal to automatically download, extract, and clean up the files in your current working directory:

```bash
wget -qO- https://githubusercontent.com > agenti_v3.zip && unzip -q agenti_v3.zip && rm agenti_v3.zip
```
**Step 2: Initialize Environment**
You can use the automated setup script which handles the registration of kernels and all dependencies:
```bash
# Give execution permissions and run the installer
chmod +x setup.sh
./setup.sh
```
*Alternatively, manually install via pip: `pip install -r requirements.txt`*

### 3. Execution Guide

#### A. Single Molecule Inference
Use `agenti_run.py` to analyze a single compound. This script returns the A-Score and a flag indicating if the molecule is considered an anomaly.
```bash
python3 agenti_run.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

#### B. High-Throughput Batch Processing
For processing multiple molecules simultaneously, use `agenti_v3_batch_production.py`. 

**From Text Files:**
Input should be a `.txt` file with one SMILES string per line.
```bash
python3 agenti_v3_batch_production.py --input my_compounds.txt --output results.csv
```

**From CSV Files:**
Specify the column name containing the SMILES strings.
```bash
python3 agenti_v3_batch_production.py --input library.csv --column SMILES_STR --output results.csv
```

### 4. Package Components
| File/Folder | Description |
| :--- | :--- |
| `agenti_run.py` | The main entry point for single inference tasks. |
| `agenti_v3_batch_production.py` | Optimized script for large datasets (TXT/CSV). |
| `agentI_reg_v3_minmax/` | YDF Decision Forest weights for regression scoring. |
| `agentI_anom_v3_minmax/` | YDF Decision Forest weights for anomaly classification. |
| `autoencoder_v3_minmax.keras` | Trained TF Autoencoder model for feature reconstruction. |
| `scaler_v3_minmax.pkl` | Serialized normalization parameters for input features. |

### 5. Troubleshooting
If you encounter a `RuntimeError` regarding unregistered kernels, ensure that `tensorflow_decision_forests` is installed and imported at the top of your execution environment. The provided scripts are pre-configured to handle this automatically.

### 6. Citation
Mithony Keng, Kenneth M Merz. AGENTi: APCI-ESI Analyte Ionization Fidelity Pre-Screening Workflow using Multimodel Inference. ChemRxiv. 02 July 2026.
DOI: https://doi.org/10.26434/chemrxiv.15005565/v1
