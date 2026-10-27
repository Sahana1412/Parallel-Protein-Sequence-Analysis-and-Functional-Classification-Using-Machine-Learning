
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

SEED = 42
BASE = "../protein_project_results_v3"

# 1. Load the dataset and recreate the split order
dataset = load_dataset("TEDBench/ted")

train_data = dataset["train"].shuffle(seed=SEED).select(range(5000))
val_data = dataset["val"].shuffle(seed=SEED).select(range(1500))
test_data = dataset["test"].shuffle(seed=SEED).select(range(1500))


def get_classes(data):
    labels = []
    for label in data["label"]:
        try:
            code = data.features["label"].int2str(int(label))
        except Exception:
            code = str(label)
        labels.append(code.split(".")[0])
    return np.array(labels)


y_train_raw = get_classes(train_data)
y_val_raw = get_classes(val_data)
y_test_raw = get_classes(test_data)

# Use the training labels to define the class mapping
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train_raw)

# Check that validation/test classes are known
unknown = (set(y_val_raw) | set(y_test_raw)) - set(encoder.classes_)
if unknown:
    raise ValueError(f"Classes not present in training data: {unknown}")

y_val = encoder.transform(y_val_raw)
y_test = encoder.transform(y_test_raw)

# 2. Load cached sequence embeddings
X_train = np.load(f"{BASE}/esm_embeddings/train_esm2_8m.npy")
X_val = np.load(f"{BASE}/esm_embeddings/validation_esm2_8m.npy")
X_test = np.load(f"{BASE}/esm_embeddings/test_esm2_8m.npy")

# 3. Verify dimensions
assert len(X_train) == len(y_train), "Training rows do not match labels"
assert len(X_val) == len(y_val), "Validation rows do not match labels"
assert len(X_test) == len(y_test), "Test rows do not match labels"

print("Feature shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Training class counts:", dict(zip(*np.unique(y_train_raw, return_counts=True))))

# 4. Tune Logistic Regression using validation macro-F1
best_model = None
best_score = -1
best_C = None

for C in [0.01, 0.1, 1, 10, 100]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)

    score = f1_score(
        y_val, val_pred, average="macro", zero_division=0
    )

    print(f"C={C:<5} Validation Macro-F1={score:.4f}")

    if score > best_score:
        best_score = score
        best_model = model
        best_C = C

# 5. Evaluate the selected model on the test set
test_pred = best_model.predict(X_test)

print("\n--- LOGISTIC REGRESSION RESULTS ---")
print("Best C:", best_C)
print(f"Validation Macro-F1: {best_score:.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
print(f"Test Balanced Accuracy: {balanced_accuracy_score(y_test, test_pred):.4f}")
print(f"Test Macro-F1: {f1_score(y_test, test_pred, average='macro', zero_division=0):.4f}")
