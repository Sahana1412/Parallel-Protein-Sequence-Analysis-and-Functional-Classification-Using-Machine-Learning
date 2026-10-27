
import pandas as pd
import numpy as np
from pathlib import Path

root = Path(r"D:\AMRITA\SEMESTER 5\AIML_BIODATA_HPC_Project")

# Check the handcrafted feature CSV and labels
features = pd.read_csv(root / "protein_project_results" / "combined_features.csv")
labels = pd.read_csv(root / "protein_project_results" / "labels.csv")

print("Combined features shape:", features.shape)
print("Labels shape:", labels.shape)
print("Feature columns:", len(features.columns))
print("Label column names:", labels.columns.tolist())
print("\nLabel counts:")
print(labels.iloc[:, 0].value_counts())

# Check saved v3 feature arrays
v3 = root / "protein_project_results_v3"

for filename in [
    "train_sequence_handcrafted.npy",
    "train_structure.npy",
    "validation_sequence_handcrafted.npy",
    "validation_structure.npy",
    "test_sequence_handcrafted.npy",
    "test_structure.npy",
    "esm_embeddings/train_esm2_8m.npy",
    "esm_embeddings/validation_esm2_8m.npy",
    "esm_embeddings/test_esm2_8m.npy",
]:
    path = v3 / filename
    if path.exists():
        array = np.load(path, mmap_mode="r")
        print(f"{filename}: {array.shape}")
    else:
        print(f"{filename}: NOT FOUND")
