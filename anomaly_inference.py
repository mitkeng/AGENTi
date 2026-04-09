
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter
import zipfile
import os
from sklearn.preprocessing import Normalizer
import ydf # Requires tensorflow_decision_forests installed
from statistics import mean

# --- Constants ---
props = {
    "H":  {"r": 1.2,  "m": 1.0,   "en": 2.2,  "ie": 13.598, "p": 0.667},
    "C":  {"r": 1.7,  "m": 12.0,  "en": 2.55, "ie": 11.26,  "p": 1.76},
    "N":  {"r": 1.55, "m": 14.0,  "en": 3.04, "ie": 14.534, "p": 1.1},
    "O":  {"r": 1.52, "m": 16.0,  "en": 3.44, "ie": 13.618, "p": 0.802},
    "P":  {"r": 1.8,  "m": 31.0,  "en": 2.19, "ie": 10.487, "p": 3.63},
    "S":  {"r": 1.8,  "m": 32.0,  "en": 2.58, "ie": 10.36,  "p": 2.9},
    "F":  {"r": 1.35, "m": 19.0,  "en": 3.98, "ie": 17.423, "p": 0.557},
    "Cl": {"r": 1.75, "m": 35.34, "en": 3.16, "ie": 12.968, "p": 2.18},
    "Br": {"r": 1.83, "m": 79.9,  "en": 2.96, "ie": 11.8,   "p": 3.05}
}

st_dev = 0.0627
norm_mean = 0.0456


def load_inference_models():
    # 1. Unzip safely
    for item in os.listdir():
        if item.endswith(".zip"):
            extract_path = item.replace(".zip", "")
            if not os.path.exists(extract_path):
                print(f"Unzipping {item}...")
                with zipfile.ZipFile(item, "r") as zip_ref:
                    zip_ref.extractall(extract_path)

    # 2. Load models

    auto_mode = keras.models.load_model("autoencoder_model.keras")
    anom_model = ydf.load_model("agentI_anom_model")
    reg_model = ydf.load_model("agentI_reg_model")
    
    # 3. Scaler Note: 
    # Normalizer() scales each sample INDIVIDUALLY to unit norm. 
    
    scaler = Normalizer() 
    
    print("Models loaded successfully.")
    return auto_mode, anom_model, reg_model, scaler

def extract_features_from_smiles(smiles_list: list) -> pd.DataFrame:
    # Extracts molecular features from a list of SMILES strings.

    feature_data = []
    for smi in smiles_list:
        try:
            s_clean = smi.strip().split(",")[0]
            counts = Counter(s_clean)

            db, tb = counts["="], counts["#"]
            halogens = counts["l"] + counts["F"] + counts["r"] + counts["I"]
            pos, neg = counts["+"], counts["-"]
            rings_from_str = sum(v for k, v in counts.items() if k.isdigit()) // 2

            r_sum = m_sum = en_sum = ie_sum = p_sum = hetero = total_atoms = 0

            for char, count in counts.items():
                if char in props:
                    p = props[char]
                    r_sum += p["r"] * count
                    m_sum += p["m"] * count
                    en_sum += p["en"] * count
                    ie_sum += p["ie"] * count
                    p_sum += p["p"] * count
                    total_atoms += count
                    if char not in "CH":
                        hetero += count
                elif char.isalpha():
                    total_atoms += count

            mol = Chem.MolFromSmiles(s_clean)
            if mol is None:
                raise ValueError("Invalid SMILES string: {}".format(s_clean))

            tpsa = round(Descriptors.TPSA(mol), 2)
            vsa = round(Descriptors.VSA_EState2(mol), 2)
            logp = round(Descriptors.MolLogP(mol), 2)
            rings_rdkit = len(Chem.GetSSSR(mol))

            rm2 = round((r_sum * total_atoms) / m_sum, 2) if m_sum else 0
            elec2 = round((ie_sum * en_sum) / (p_sum * total_atoms), 2) if (p_sum * total_atoms) else 0

            feature_data.append([
                s_clean, rm2, elec2, db, tb, hetero, halogens, int(rings_rdkit),
                tpsa, vsa, logp, pos, neg
            ])
        except Exception as e:
            print("Error processing SMILES {}: {}".format(smi, e))
            continue

    headers = [
        "smile", "radii_mass2", "electronic2", "double_bond2", "triple_bond2",
        "hetero2", "halogen", "ring2", "TPSA", "VSA", "logP", "pos_charge", "neg_charge"
    ]
    df_features = pd.DataFrame(feature_data, columns=headers)
    return df_features.drop(columns=["smile"])

def predict_anomalies(smiles_list: list, auto_mode, anom_model, reg_model, scaler, threshold: float = 1.40) -> pd.DataFrame:
    # Predicts anomaly scores for a list of SMILES strings.

    original_smiles_df = pd.DataFrame({"smile": smiles_list})
    df_features = extract_features_from_smiles(smiles_list)

    if df_features.empty:
        return pd.DataFrame(columns=["SMILE", "A-Score", "Is Anomalous"])

    scaled_data = scaler.transform(df_features.values)

    reconstruction_errors = np.mean(np.power(scaled_data - auto_mode.predict(scaled_data), 2), axis=1)

    prediction1 = anom_model.predict(df_features)
    prediction2 = reg_model.predict(df_features)

    results = []
    for i in range(len(df_features)): # Iterate through the processed features
        current_smile = original_smiles_df.iloc[i]["smile"]
        a_score = mean([prediction1[i], prediction2[i]]) + reconstruction_errors[i] / st_dev
        is_anomalous = "Yes" if a_score > threshold else "No"
        results.append({
            "SMILE": current_smile,
            "A-Score": round(a_score, 2),
            "Is Anomalous": is_anomalous,
            "Threshold": threshold
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Loading models...")
    auto_mode, anom_model, reg_model, scaler = load_inference_models()
    print("Models loaded and ready.")

    # Example usage from a list of SMILES
    example_smiles = [
        "CN(C)CCN(C)C1=CC(=C(C=C1NC(=O)C=C)NC2=NC=CC(=N2)C3=CN(C4=CC=CC=C43)C5CC5)OC",
        "CCO",
        "C1CCCCC1",
        "CC(=O)Oc1ccccc1C(=O)O"
    ]
    print("Predicting anomalies for example SMILES list ---")
    api_results_df = predict_anomalies(example_smiles, auto_mode, anom_model, reg_model, scaler)
    print(api_results_df.to_string(index=False))

    # Example usage from a CSV file (assuming 'smiles_input.csv' exists with a 'SMILES_String' column)
    csv_file_path = "smiles_input.csv"
    if os.path.exists(csv_file_path):

        print("Predicting anomalies from {{{{}}}} ---".format(csv_file_path))
        df_smiles_from_csv = pd.read_csv(csv_file_path)
        smiles_list_from_csv = df_smiles_from_csv["SMILES_String"].tolist()
        csv_api_results_df = predict_anomalies(smiles_list_from_csv, auto_mode, anom_model, reg_model, scaler)
        print(csv_api_results_df.to_string(index=False))
    else:

        print("Warning: {{{{}}}} not found. Skipping CSV prediction example.".format(csv_file_path))
