[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) ![user](https://img.shields.io/badge/GoogleColab-grey?style=flat&logo=googlecolab) ![user](https://img.shields.io/badge/Chemodeling-App-yellow?) ![user](https://img.shields.io/badge/Userfriend-1.0-sgreen?) 

## 🧪 AGENTi: Probing Ionization Fidelity

<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/ee2f391f-b20b-45d9-b3aa-8e24fd4f4a40" />

### We try, so the community can optimize.
---

## 🚀 How to Use AGENTi

### 1. Choose Your Input
You can feed the model data in two ways:
*   **Single SMILES**: Provide a single string for a quick check (e.g., `CC(=O)Oc1ccccc1C(=O)O` for Aspirin).
*   **Batch CSV**: Upload a `.csv` file containing a column named `SMILES_String` to process hundreds of molecules at once.

### 2. Automated Processing
Once you run the script, the system performs these background steps:
*   **Feature Extraction**: Converts the SMILES into 13 mathematical descriptors (like `VSA` and `TPSA`).
*   **Neural Analysis**: Runs the data through three specialized models: an **Autoencoder**, an **Anomaly Model**, and a **Regression Model**.
*   **Scoring**: Combines the model outputs into a final **A-Score**.

### 3. Review the Output
The script generates a clean table with three key columns:
*   **SMILE**: The molecule identifier.
*   **A-Score**: The calculated risk value.
*   **Is Anomalous**: A simple **Yes/No** based on whether the score exceeds your **1.40 threshold**.

---

## 🛠️ Installation & Execution

### **Quick Start (Linux/macOS)**
Run the following commands in your terminal to set up the environment and run a test on Aspirin:

```bash
# Clone the repository
git clone https://github.com/mitkeng/AGENTi.git
cd AGENTi

# Install dependencies and download models
chmod +x setup.sh
./setup.sh

# Run the inference script
python3 run_test.py
