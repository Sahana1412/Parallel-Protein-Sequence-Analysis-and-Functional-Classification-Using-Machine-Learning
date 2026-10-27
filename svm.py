
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "protein_project_results_v3"
EMBEDDINGS = RESULTS / "esm_embeddings"
OUT = Path(__file__).resolve().parent / "svm_results"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_SIZE = 5000
VAL_SIZE = 1500
TEST_SIZE = 1500
SEED = 42
C_VALUES = [0.1, 1.0, 10.0]


# --------------------------------------------------
# LABEL DECODING
# --------------------------------------------------

def decode_class(split, label):
    try:
        value = split.features["label"].int2str(int(label))
    except Exception:
        value = str(label)

    return str(value).split(".")[0]


def load_labels():
    print("Loading TEDBench/ted...")
    dataset = load_dataset("TEDBench/ted")

    splits = {
        "train": dataset["train"].shuffle(seed=SEED).select(
            range(min(TRAIN_SIZE, len(dataset["train"])))
        ),
        "validation": dataset["val"].shuffle(seed=SEED).select(
            range(min(VAL_SIZE, len(dataset["val"])))
        ),
        "test": dataset["test"].shuffle(seed=SEED).select(
            range(min(TEST_SIZE, len(dataset["test"])))
        ),
    }

    labels = {}

    for name, split in splits.items():
        labels[name] = np.array(
            [decode_class(split, row["label"]) for row in split],
            dtype=str,
        )
        print(
            f"{name}: {len(labels[name])} samples; "
            f"classes={dict(pd.Series(labels[name]).value_counts())}"
        )

    return labels


# --------------------------------------------------
# FEATURE LOADING
# --------------------------------------------------

def load_features(path, expected_rows):
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    x = np.load(path)

    if x.ndim != 2:
        raise ValueError(f"{path.name} is not a 2D array: {x.shape}")

    if x.shape[0] != expected_rows:
        raise ValueError(
            f"{path.name}: {x.shape[0]} rows but "
            f"{expected_rows} labels. Stopping."
        )

    if not np.isfinite(x).all():
        raise ValueError(f"{path.name} contains NaN or infinite values.")

    print(f"Loaded {path.name}: {x.shape}")
    return x.astype(np.float32, copy=False)


# --------------------------------------------------
# MAIN SVM EXPERIMENT
# --------------------------------------------------

def main():
    start = time.perf_counter()

    print("=" * 65)
    print("CATH CLASSIFICATION USING SVM")
    print("=" * 65)

    labels = load_labels()

    X = {}

    files = {
        "train": (
            EMBEDDINGS / "train_esm2_8m.npy",
            RESULTS / "train_structure.npy",
        ),
        "validation": (
            EMBEDDINGS / "validation_esm2_8m.npy",
            RESULTS / "validation_structure.npy",
        ),
        "test": (
            EMBEDDINGS / "test_esm2_8m.npy",
            RESULTS / "test_structure.npy",
        ),
    }

    for split, (esm_file, struct_file) in files.items():
        n = len(labels[split])
        X[split] = {
            "sequence": load_features(esm_file, n),
            "structure": load_features(struct_file, n),
        }

        if X[split]["sequence"].shape[1] != 320:
            raise ValueError("Expected 320 ESM-2 features.")

        if X[split]["structure"].shape[1] != 28:
            raise ValueError("Expected 28 structural features.")

    # Encode labels using training classes only.
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(labels["train"])

    val_mask = np.isin(labels["validation"], encoder.classes_)
    test_mask = np.isin(labels["test"], encoder.classes_)

    if not val_mask.all():
        print("Excluding validation labels unseen during training.")

    if not test_mask.all():
        print("Excluding test labels unseen during training.")

    y_val = encoder.transform(labels["validation"][val_mask])
    y_test = encoder.transform(labels["test"][test_mask])

    feature_sets = {}

    for feature in ["sequence", "structure"]:
        feature_sets[feature] = (
            X["train"][feature],
            X["validation"][feature][val_mask],
            X["test"][feature][test_mask],
        )

    feature_sets["combined"] = (
        np.concatenate(
            [X["train"]["sequence"], X["train"]["structure"]], axis=1
        ),
        np.concatenate(
            [
                X["validation"]["sequence"][val_mask],
                X["validation"]["structure"][val_mask],
            ],
            axis=1,
        ),
        np.concatenate(
            [
                X["test"]["sequence"][test_mask],
                X["test"]["structure"][test_mask],
            ],
            axis=1,
        ),
    )

    results = []
    reports = []

    # --------------------------------------------------
    # Train linear and RBF SVMs for each feature set.
    # Select C using validation macro-F1.
    # --------------------------------------------------

    total = len(feature_sets) * 2
    count = 0

    for feature_name, (xtr, xval, xtest) in feature_sets.items():
        for kernel in ["linear", "rbf"]:
            count += 1
            print(f"\n[{count}/{total}] {feature_name} | {kernel} SVM")

            best_model = None
            best_c = None
            best_val_f1 = -1

            for c in C_VALUES:
                model = make_pipeline(
                    StandardScaler(),
                    SVC(
                        kernel=kernel,
                        C=c,
                        class_weight="balanced",
                        cache_size=1000,
                    ),
                )

                t0 = time.perf_counter()
                model.fit(xtr, y_train)
                pred_val = model.predict(xval)

                val_f1 = f1_score(
                    y_val, pred_val, average="macro", zero_division=0
                )

                print(
                    f"  C={c}: validation macro-F1={val_f1:.4f}; "
                    f"time={time.perf_counter() - t0:.1f}s"
                )

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model = model
                    best_c = c

            # Evaluate the selected model once on the test set.
            pred_test = best_model.predict(xtest)

            result = {
                "features": feature_name,
                "kernel": kernel,
                "best_C": best_c,
                "validation_macro_f1": best_val_f1,
                "test_accuracy": accuracy_score(y_test, pred_test),
                "test_balanced_accuracy": balanced_accuracy_score(
                    y_test, pred_test
                ),
                "test_macro_f1": f1_score(
                    y_test, pred_test,
                    average="macro", zero_division=0
                ),
                "test_weighted_f1": f1_score(
                    y_test, pred_test,
                    average="weighted", zero_division=0
                ),
                "train_samples": len(y_train),
                "validation_samples": len(y_val),
                "test_samples": len(y_test),
                "feature_count": xtr.shape[1],
            }
            results.append(result)

            print(
                f"  TEST accuracy={result['test_accuracy']:.4f}, "
                f"balanced accuracy={result['test_balanced_accuracy']:.4f}, "
                f"macro-F1={result['test_macro_f1']:.4f}, "
                f"weighted-F1={result['test_weighted_f1']:.4f}"
            )

            report = classification_report(
                y_test,
                pred_test,
                labels=np.arange(len(encoder.classes_)),
                target_names=[str(c) for c in encoder.classes_],
                zero_division=0,
            )

            reports.append(
                f"FEATURES: {feature_name}\n"
                f"KERNEL: {kernel}\n"
                f"BEST C: {best_c}\n\n{report}\n"
            )

            cm = confusion_matrix(
                y_test,
                pred_test,
                labels=np.arange(len(encoder.classes_)),
            )

            pd.DataFrame(
                cm,
                index=[f"actual_{c}" for c in encoder.classes_],
                columns=[f"predicted_{c}" for c in encoder.classes_],
            ).to_csv(
                OUT / f"confusion_{feature_name}_{kernel}.csv"
            )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    df = pd.DataFrame(results).sort_values(
        "test_macro_f1", ascending=False
    )

    df.to_csv(OUT / "svm_comparison.csv", index=False)

    (OUT / "classification_reports.txt").write_text(
        "\n\n".join(reports), encoding="utf-8"
    )

    (OUT / "run_info.json").write_text(
        json.dumps(
            {
                "dataset": "TEDBench/ted",
                "target": "CATH Class",
                "seed": SEED,
                "classes": encoder.classes_.tolist(),
                "C_values": C_VALUES,
                "elapsed_seconds": round(time.perf_counter() - start, 2),
                "feature_dimensions": {
                    "sequence": 320,
                    "structure": 28,
                    "combined": 348,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("FINAL SVM COMPARISON")
    print("=" * 65)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nResults saved in:", OUT)
    print(" - svm_comparison.csv")
    print(" - classification_reports.txt")
    print(" - run_info.json")
    print(" - confusion matrix CSV files")
    print(f"\nTotal time: {time.perf_counter() - start:.1f} seconds")


if __name__ == "__main__":
    main()
