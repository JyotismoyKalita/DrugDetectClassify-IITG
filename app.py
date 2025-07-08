import streamlit as st
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from Mold2_pywrapper import Mold2

# --- File Paths ---
PATH_MOLD2_ZIP = "Drug Detection/Tools/Mold2-Executable-File.zip"
DETECTION_MODEL_PATH = "Drug Detection/Model/druglikeness_logreg_bundle.pkl"
CLASSIFICATION_MODEL_PATH = "Drug Classification/Model/drugclass_logreg_bundle.pkl"
CLASS_MAPPING_PATH = "Drug Classification/Dataset/atc/dataset.csv"

# --- Constants ---
ECFP_BITS = 2048
MACCS_BITS = 166

# --- Load Models ---
@st.cache_resource(show_spinner=False)
def load_detection_model():
    return joblib.load(DETECTION_MODEL_PATH)

@st.cache_resource(show_spinner=False)
def load_classification_model():
    return joblib.load(CLASSIFICATION_MODEL_PATH)["pipeline"]

@st.cache_resource(show_spinner=False)
def load_atc_mapping():
    df = pd.read_csv(CLASS_MAPPING_PATH)
    return dict(zip(df["atc_numeric"], df["atc_level1"]))

model_bundle = load_detection_model()
detection_pipeline = model_bundle["pipeline"]
detection_threshold = model_bundle["threshold"]
classification_pipeline = load_classification_model()
num_to_atc = load_atc_mapping()

# --- Feature Extraction ---
def fp_to_array(fp, n_bits):
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def extract_all_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        # MolD2
        mold2 = Mold2.from_executable(PATH_MOLD2_ZIP)
        mold2_df = pd.DataFrame(mold2.calculate([mol]))

        # ECFP4
        morgan_gen = GetMorganGenerator(radius=2, fpSize=ECFP_BITS)
        ecfp = morgan_gen.GetFingerprint(mol)
        ecfp_array = fp_to_array(ecfp, ECFP_BITS)
        ecfp_df = pd.DataFrame([ecfp_array], columns=[f"ECFP4_{i}" for i in range(ECFP_BITS)])

        # MACCS
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_array = fp_to_array(maccs, MACCS_BITS + 1)[1:]  # Drop bit 0
        maccs_df = pd.DataFrame([maccs_array], columns=[f"MACCS_{i}" for i in range(MACCS_BITS)])

        combined = pd.concat([mold2_df, ecfp_df, maccs_df], axis=1)
        return combined, ecfp_array  # Return both for Model 1 and Model 2

    except Exception as e:
        st.error(f"[Feature extraction error] {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Drug Detection & ATC Classification", page_icon="🧪", layout="centered")
st.title("🧪 Drug Detection and ATC Level‑1 Classification")

st.markdown("Enter a **SMILES** string. The app will:")
st.markdown("- Detect if the compound is Drug-like")
st.markdown("- If **Drug**, predict ATC Level‑1 class using Logistic Regression")

smiles = st.text_input("Enter SMILES", "CC(=O)Oc1ccccc1C(=O)O")  # aspirin default
min_prob_threshold = st.slider(
    "Minimum confidence to display ATC code", 
    min_value=0.01, 
    max_value=1.0, 
    value=0.10, 
    step=0.01
)

if st.button("🔍 Analyze Molecule"):

    feats_combined = extract_all_features(smiles)
    if feats_combined is None:
        st.error("❌ Invalid or unprocessable SMILES.")
        st.stop()

    features_full, ecfp_arr = feats_combined

    # --- Drug Detection ---
    try:
        proba = detection_pipeline.predict_proba(features_full)[0, 1]
        is_drug = int(proba >= detection_threshold)
    except Exception as e:
        st.error(f"⚠️ Drug detection failed: {str(e)}")
        st.stop()

    if is_drug:
        st.success(f"✅ Detected as Drug (Probability: {proba:.3f} ≥ Threshold: {detection_threshold})")

        # Predict ATC Level 1
        try:
            x_query = pd.DataFrame([ecfp_arr], columns=[f"ECFP4_{i}" for i in range(ECFP_BITS)])
            probs = classification_pipeline.predict_proba(x_query)[0]
            results = pd.DataFrame({
                "ATC Code": [num_to_atc[i] for i in range(len(probs))],
                "Probability": probs
            }).sort_values("Probability", ascending=False).reset_index(drop=True)

            filtered_results = results[results["Probability"] >= min_prob_threshold]

            if filtered_results.empty:
                st.warning("⚠️ No ATC codes predicted with sufficient confidence.")
            else:
                st.subheader("Predicted ATC Level‑1 Codes:")
                st.table(filtered_results[["ATC Code"]])

            st.download_button("📥 Download Full Prediction Table",
                               data=results.to_csv(index=False).encode(),
                               file_name="atc_predictions.csv")
        except Exception as e:
            st.error(f"⚠️ ATC prediction failed: {str(e)}")

    else:
        st.warning(f"❌ Detected as Non-Drug (Probability: {proba:.3f} < Threshold: {detection_threshold})")
