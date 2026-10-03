import os
import time
import json
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_SIZE = 5000
VAL_SIZE = 1500
TEST_SIZE = 1500

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# Benchmark these CPU configurations
THREAD_COUNTS = [1, 2, 4, 8, 12]

# ESM batch size
ESM_BATCH_SIZE = 2

# MLP
HIDDEN_1 = 256
HIDDEN_2 = 128

EPOCHS = 40
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

PATIENCE = 7

SEED = 42

CACHE_DIR = r"D:\HuggingFaceCache"

OUTPUT_DIR = "hpc_cpu_results_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# THREAD CONTROL
# ============================================================

def set_cpu_threads(n):

    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)

    torch.set_num_threads(n)

    try:
        torch.set_num_interop_threads(
            min(2, n)
        )
    except RuntimeError:
        pass


# ============================================================
# SYSTEM INFORMATION
# ============================================================

print("=" * 75)
print("SEQUENTIAL CPU vs OPENMP CPU BENCHMARK")
print("=" * 75)

print(
    "PyTorch:",
    torch.__version__
)

print(
    "CPU threads available:",
    os.cpu_count()
)

print(
    "OpenMP backend:",
    torch.backends.openmp.is_available()
)

print(
    "MKL:",
    torch.backends.mkl.is_available()
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading TEDBench/ted...")

dataset = load_dataset(
    "TEDBench/ted",
    cache_dir=CACHE_DIR
)

train = dataset["train"].shuffle(
    seed=SEED
).select(
    range(TRAIN_SIZE)
)

val = dataset["val"].shuffle(
    seed=SEED
).select(
    range(VAL_SIZE)
)

test = dataset["test"].shuffle(
    seed=SEED
).select(
    range(TEST_SIZE)
)

print(
    "Train:",
    len(train)
)

print(
    "Validation:",
    len(val)
)

print(
    "Test:",
    len(test)
)


# ============================================================
# CATH CLASS LABEL
# ============================================================

def decode_class(data, label):

    try:

        code = data.features[
            "label"
        ].int2str(
            int(label)
        )

    except Exception:

        code = str(label)

    return code.split(".")[0]


y_train_text = np.array([
    decode_class(train, x)
    for x in train["label"]
])

y_val_text = np.array([
    decode_class(val, x)
    for x in val["label"]
])

y_test_text = np.array([
    decode_class(test, x)
    for x in test["label"]
])


# ============================================================
# ENCODE CATH CLASS
# ============================================================

encoder = LabelEncoder()

y_train = encoder.fit_transform(
    y_train_text
)

print(
    "\nCATH Class labels:",
    list(encoder.classes_)
)

# Keep only classes present in training
val_mask = np.isin(
    y_val_text,
    encoder.classes_
)

test_mask = np.isin(
    y_test_text,
    encoder.classes_
)

y_val = encoder.transform(
    y_val_text[val_mask]
)

y_test = encoder.transform(
    y_test_text[test_mask]
)

print(
    "Validation usable:",
    val_mask.sum(),
    "/",
    len(val_mask)
)

print(
    "Test usable:",
    test_mask.sum(),
    "/",
    len(test_mask)
)


# ============================================================
# ESM-2
# ============================================================

print("\nLoading ESM-2...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)

esm = AutoModel.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)

esm.eval()

print("ESM-2 loaded.")


# ============================================================
# ESM EMBEDDINGS
# ============================================================

def generate_embeddings(data):

    embeddings = []

    start = time.perf_counter()

    with torch.no_grad():

        for start_idx in range(
            0,
            len(data),
            ESM_BATCH_SIZE
        ):

            batch = data[
                start_idx:
                start_idx + ESM_BATCH_SIZE
            ]

            sequences = batch[
                "sequence"
            ]

            tokens = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )

            output = esm(
                **tokens
            )

            hidden = (
                output.last_hidden_state
            )

            mask = (
                tokens["attention_mask"]
                .unsqueeze(-1)
            )

            pooled = (
                hidden * mask
            ).sum(
                dim=1
            ) / mask.sum(
                dim=1
            ).clamp(min=1)

            embeddings.append(
                pooled.numpy()
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        np.vstack(
            embeddings
        ).astype(
            np.float32
        ),
        elapsed
    )


# ============================================================
# STRUCTURAL FEATURES
# ============================================================

def structural_features(coords):

    coords = np.asarray(
        coords,
        dtype=np.float32
    )

    if (
        coords.ndim != 3
        or coords.shape[1:] != (3, 3)
    ):
        return np.zeros(
            28,
            dtype=np.float32
        )

    length = coords.shape[0]

    if length < 2:
        return np.zeros(
            28,
            dtype=np.float32
        )

    N = coords[:, 0, :]
    CA = coords[:, 1, :]
    C = coords[:, 2, :]

    ca_bonds = np.linalg.norm(
        CA[1:] - CA[:-1],
        axis=1
    )

    ca_mean = np.mean(ca_bonds)
    ca_std = np.std(ca_bonds)
    ca_min = np.min(ca_bonds)
    ca_max = np.max(ca_bonds)
    ca_q25 = np.percentile(
        ca_bonds, 25
    )
    ca_median = np.median(ca_bonds)
    ca_q75 = np.percentile(
        ca_bonds, 75
    )

    n_ca = np.linalg.norm(
        N - CA,
        axis=1
    )

    ca_c = np.linalg.norm(
        CA - C,
        axis=1
    )

    c_n = np.linalg.norm(
        C[:-1] - N[1:],
        axis=1
    )

    n_ca_mean = np.mean(n_ca)
    ca_c_mean = np.mean(ca_c)
    c_n_mean = np.mean(c_n)

    center = np.mean(
        CA,
        axis=0
    )

    radial = np.linalg.norm(
        CA - center,
        axis=1
    )

    radius_of_gyration = np.sqrt(
        np.mean(radial ** 2)
    )

    end_to_end = np.linalg.norm(
        CA[-1] - CA[0]
    )

    x_std = np.std(CA[:, 0])
    y_std = np.std(CA[:, 1])
    z_std = np.std(CA[:, 2])

    squared_norms = np.sum(
        CA * CA,
        axis=1
    )

    contact_count = 0
    pairwise_sum = 0.0
    pairwise_sum_sq = 0.0
    pairwise_count = 0
    max_pairwise = 0.0
    contact_separation_sum = 0.0

    distance_samples = []

    for i in range(
        length - 1
    ):

        remaining = CA[i + 1:]

        dot_products = np.dot(
            remaining,
            CA[i]
        )

        distance_squared = (
            squared_norms[i]
            + squared_norms[i + 1:]
            - 2.0 * dot_products
        )

        distance_squared = np.maximum(
            distance_squared,
            0.0
        )

        distances = np.sqrt(
            distance_squared
        )

        pairwise_sum += np.sum(
            distances
        )

        pairwise_sum_sq += np.sum(
            distances ** 2
        )

        pairwise_count += len(
            distances
        )

        max_pairwise = max(
            max_pairwise,
            np.max(distances)
        )

        if len(distance_samples) < 5000:

            take = min(
                5000 - len(distance_samples),
                len(distances)
            )

            if take > 0:

                distance_samples.extend(
                    distances[:take].tolist()
                )

        contacts = (
            distances < 8.0
        )

        n_contacts = np.sum(
            contacts
        )

        contact_count += n_contacts

        if n_contacts > 0:

            j_indices = np.arange(
                i + 1,
                length
            )

            separations = (
                j_indices[contacts] - i
            )

            contact_separation_sum += (
                np.sum(separations)
            )

    possible_contacts = (
        length * (length - 1) / 2
    )

    contact_density = (
        contact_count
        / possible_contacts
    )

    if contact_count > 0:

        contact_order = (
            contact_separation_sum
            / contact_count
            / length
        )

    else:

        contact_order = 0.0

    mean_pairwise = (
        pairwise_sum
        / pairwise_count
    )

    variance = (
        pairwise_sum_sq
        / pairwise_count
        - mean_pairwise ** 2
    )

    variance = max(
        variance,
        0.0
    )

    std_pairwise = np.sqrt(
        variance
    )

    if distance_samples:

        distance_samples = np.asarray(
            distance_samples,
            dtype=np.float32
        )

        pair_q25 = np.percentile(
            distance_samples,
            25
        )

        pair_median = np.percentile(
            distance_samples,
            50
        )

        pair_q75 = np.percentile(
            distance_samples,
            75
        )

    else:

        pair_q25 = 0.0
        pair_median = 0.0
        pair_q75 = 0.0

    compactness = (
        radius_of_gyration
        / max(end_to_end, 1e-6)
    )

    return np.array(
        [
            contact_count,
            contact_density,
            contact_order,
            radius_of_gyration,
            end_to_end,
            ca_mean,
            ca_std,
            ca_min,
            ca_max,
            ca_q25,
            ca_median,
            ca_q75,
            n_ca_mean,
            ca_c_mean,
            c_n_mean,
            mean_pairwise,
            std_pairwise,
            max_pairwise,
            pair_q25,
            pair_median,
            pair_q75,
            x_std,
            y_std,
            z_std,
            compactness,
            length,
            np.mean(radial),
            np.std(radial)
        ],
        dtype=np.float32
    )


def extract_structures(data):

    result = []

    start = time.perf_counter()

    for i in range(len(data)):

        result.append(
            structural_features(
                data[i]["coords"]
            )
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        np.vstack(result),
        elapsed
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 75)
print("FEATURE EXTRACTION")
print("=" * 75)

print("\nESM-2 embeddings...")

seq_train, esm_train_time = (
    generate_embeddings(train)
)

seq_val, esm_val_time = (
    generate_embeddings(val)
)

seq_test, esm_test_time = (
    generate_embeddings(test)
)

esm_total_time = (
    esm_train_time
    + esm_val_time
    + esm_test_time
)

print(
    "ESM total:",
    round(esm_total_time, 2),
    "seconds"
)


print("\nStructural features...")

struct_train, struct_train_time = (
    extract_structures(train)
)

struct_val, struct_val_time = (
    extract_structures(val)
)

struct_test, struct_test_time = (
    extract_structures(test)
)

struct_total_time = (
    struct_train_time
    + struct_val_time
    + struct_test_time
)

print(
    "Structural total:",
    round(struct_total_time, 2),
    "seconds"
)


# ============================================================
# COMBINE FEATURES
# ============================================================

X_train = np.hstack(
    [
        seq_train,
        struct_train
    ]
)

X_val = np.hstack(
    [
        seq_val[val_mask],
        struct_val[val_mask]
    ]
)

X_test = np.hstack(
    [
        seq_test[test_mask],
        struct_test[test_mask]
    ]
)

print(
    "\nFeature dimensions:",
    X_train.shape
)


# ============================================================
# NORMALIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
).astype(
    np.float32
)

X_val = scaler.transform(
    X_val
).astype(
    np.float32
)

X_test = scaler.transform(
    X_test
).astype(
    np.float32
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train
)

class_weights = (
    len(y_train)
    / (
        len(class_counts)
        * class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)


# ============================================================
# MLP MODEL
# ============================================================

class ProteinMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_1
            ),

            nn.BatchNorm1d(
                HIDDEN_1
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                HIDDEN_1,
                HIDDEN_2
            ),

            nn.BatchNorm1d(
                HIDDEN_2
            ),

            nn.ReLU(),

            nn.Dropout(
                0.15
            ),

            nn.Linear(
                HIDDEN_2,
                num_classes
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    threads,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    set_cpu_threads(
        threads
    )

    torch.manual_seed(
        SEED
    )

    model = ProteinMLP(
        X_train.shape[1],
        len(
            np.unique(y_train)
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    Xtr = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    ytr = torch.tensor(
        y_train,
        dtype=torch.long
    )

    Xv = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    yv = torch.tensor(
        y_val,
        dtype=torch.long
    )

    Xte = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    best_val = -1.0
    best_state = None
    patience_counter = 0

    start = time.perf_counter()

    for epoch in range(
        EPOCHS
    ):

        model.train()

        optimizer.zero_grad()

        output = model(
            Xtr
        )

        loss = criterion(
            output,
            ytr
        )

        loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():

            val_output = model(
                Xv
            )

            val_pred = (
                val_output
                .argmax(
                    dim=1
                )
                .numpy()
            )

        val_acc = accuracy_score(
            y_val,
            val_pred
        )

        scheduler.step(
            val_acc
        )

        if val_acc > best_val:

            best_val = val_acc

            best_state = {
                k: v.clone()
                for k, v
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    training_time = (
        time.perf_counter()
        - start
    )

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():

        predictions = (
            model(Xte)
            .argmax(
                dim=1
            )
            .numpy()
        )

    test_accuracy = accuracy_score(
        y_test,
        predictions
    )

    test_balanced = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    test_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    return {
        "training_time_seconds":
            training_time,

        "epochs_completed":
            epoch + 1,

        "validation_accuracy":
            best_val,

        "test_accuracy":
            test_accuracy,

        "test_balanced_accuracy":
            test_balanced,

        "test_macro_f1":
            test_f1
    }


# ============================================================
# HPC BENCHMARK
# ============================================================

print("\n" + "=" * 75)
print("STARTING SEQUENTIAL / OPENMP BENCHMARK")
print("=" * 75)

results = []

baseline_total = None


for threads in THREAD_COUNTS:

    print("\n" + "-" * 75)

    if threads == 1:

        mode = "Sequential CPU"

    else:

        mode = (
            f"OpenMP CPU ({threads} threads)"
        )

    print(
        mode
    )

    set_cpu_threads(
        threads
    )

    start_total = time.perf_counter()

    result = train_model(
        threads,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )

    total_time = (
        time.perf_counter()
        - start_total
    )

    if baseline_total is None:

        baseline_total = total_time

    speedup = (
        baseline_total
        / total_time
    )

    efficiency = (
        speedup / threads
    )

    row = {
        "mode": mode,
        "threads": threads,
        "training_time_seconds":
            result[
                "training_time_seconds"
            ],
        "total_time_seconds":
            total_time,
        "speedup":
            speedup,
        "efficiency":
            efficiency,
        "epochs":
            result[
                "epochs_completed"
            ],
        "validation_accuracy":
            result[
                "validation_accuracy"
            ],
        "test_accuracy":
            result[
                "test_accuracy"
            ],
        "test_balanced_accuracy":
            result[
                "test_balanced_accuracy"
            ],
        "test_macro_f1":
            result[
                "test_macro_f1"
            ]
    }

    results.append(
        row
    )

    print(
        f"Training time: "
        f"{result['training_time_seconds']:.3f} s"
    )

    print(
        f"Total time: "
        f"{total_time:.3f} s"
    )

    print(
        f"Speedup: "
        f"{speedup:.3f}x"
    )

    print(
        f"Efficiency: "
        f"{efficiency * 100:.2f}%"
    )

    print(
        f"Validation accuracy: "
        f"{result['validation_accuracy']:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{result['test_accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{result['test_macro_f1']:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(
    results
)

csv_path = os.path.join(
    OUTPUT_DIR,
    "cpu_openmp_benchmark.csv"
)

df.to_csv(
    csv_path,
    index=False
)


config = {

    "dataset":
        "TEDBench/ted",

    "train":
        TRAIN_SIZE,

    "validation":
        VAL_SIZE,

    "test":
        TEST_SIZE,

    "esm_model":
        MODEL_NAME,

    "sequence_features":
        320,

    "structural_features":
        28,

    "combined_features":
        348,

    "classifier":
        "2-layer MLP",

    "hidden_layers":
        [
            HIDDEN_1,
            HIDDEN_2
        ],

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "feature_extraction_esm_seconds":
        esm_total_time,

    "feature_extraction_structure_seconds":
        struct_total_time
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "config.json"
    ),
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("FINAL CPU + OPENMP RESULTS")
print("=" * 75)

print(
    df.to_string(
        index=False
    )
)

print(
    "\nSaved:",
    csv_path
)

print("\nDone.")