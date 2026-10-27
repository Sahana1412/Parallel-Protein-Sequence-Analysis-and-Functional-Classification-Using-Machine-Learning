
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon

SEED = 42
BASE = "../protein_project_results_v3"

# --------------------------------------------------
# 1. Load labels in the same order as the feature data
# --------------------------------------------------
print("Loading TEDBench labels...")

dataset = load_dataset("TEDBench/ted")

train_data = (
    dataset["train"]
    .shuffle(seed=SEED)
    .select(range(5000))
)

def get_classes(data):
    result = []
    for label in data["label"]:
        try:
            code = data.features["label"].int2str(int(label))
        except Exception:
            code = str(label)
        result.append(code.split(".")[0])
    return np.array(result)

y_raw = get_classes(train_data)
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

print("Classes:", encoder.classes_)
print("Samples:", len(y))

# --------------------------------------------------
# 2. Load cached features
# --------------------------------------------------
print("\nLoading cached features...")

X_seq = np.load(
    f"{BASE}/esm_embeddings/train_esm2_8m.npy"
)
X_struct = np.load(
    f"{BASE}/train_structure.npy"
)

assert len(X_seq) == len(y), (
    "Sequence feature count does not match labels."
)
assert len(X_struct) == len(y), (
    "Structural feature count does not match labels."
)

X_combined = np.hstack([X_seq, X_struct])

feature_sets = {
    "Sequence": X_seq,
    "Structure": X_struct,
    "Sequence + Structure": X_combined
}

# --------------------------------------------------
# 3. Define linear and nonlinear models
# --------------------------------------------------
def make_models():
    return {
        "Linear Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED
            )
        ),
        "Nonlinear RBF SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced"
            )
        )
    }

# --------------------------------------------------
# 4. Repeated stratified cross-validation
# --------------------------------------------------
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=SEED
)

all_results = []
summary = []

for feature_name, X in feature_sets.items():
    print(f"\nTesting: {feature_name}")

    scores = {
        "Linear Logistic Regression": [],
        "Nonlinear RBF SVM": []
    }

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        models = make_models()

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            score = f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0
            )

            scores[model_name].append(score)

            all_results.append({
                "Features": feature_name,
                "Fold": fold,
                "Model": model_name,
                "Macro-F1": score
            })

    linear = np.array(
        scores["Linear Logistic Regression"]
    )
    nonlinear = np.array(
        scores["Nonlinear RBF SVM"]
    )

    differences = nonlinear - linear

    print(
        f"Linear mean macro-F1: "
        f"{linear.mean():.4f} +/- {linear.std(ddof=1):.4f}"
    )
    print(
        f"RBF mean macro-F1: "
        f"{nonlinear.mean():.4f} +/- "
        f"{nonlinear.std(ddof=1):.4f}"
    )
    print(
        f"Mean improvement (RBF - linear): "
        f"{differences.mean():+.4f}"
    )

    # Paired test; exploratory because CV folds overlap.
    if np.allclose(differences, 0):
        p_value = 1.0
    else:
        _, p_value = wilcoxon(
            nonlinear,
            linear,
            alternative="greater",
            method="approx"
        )

    print(f"Wilcoxon p-value (exploratory): {p_value:.4f}")

    if differences.mean() > 0 and p_value < 0.05:
        conclusion = (
            "Evidence that RBF SVM improves macro-F1; "
            "not proof of global nonlinearity."
        )
    elif differences.mean() > 0:
        conclusion = (
            "RBF mean is higher, but improvement is "
            "not statistically convincing in this test."
        )
    else:
        conclusion = (
            "No evidence that RBF SVM improves over "
            "the linear baseline."
        )

    print("Interpretation:", conclusion)

    summary.append({
        "Features": feature_name,
        "Linear mean Macro-F1": linear.mean(),
        "Linear SD": linear.std(ddof=1),
        "RBF mean Macro-F1": nonlinear.mean(),
        "RBF SD": nonlinear.std(ddof=1),
        "Mean improvement": differences.mean(),
        "Exploratory p-value": p_value,
        "Interpretation": conclusion
    })

# --------------------------------------------------
# 5. Save all results
# --------------------------------------------------
pd.DataFrame(all_results).to_csv(
    "nonlinearity_fold_results.csv", index=False
)

pd.DataFrame(summary).to_csv(
    "nonlinearity_summary.csv", index=False
)

print("\nSaved:")
print("nonlinearity_fold_results.csv")
print("nonlinearity_summary.csv")
