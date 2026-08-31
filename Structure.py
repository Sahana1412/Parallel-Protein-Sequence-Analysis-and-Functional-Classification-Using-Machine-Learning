# ============================================================
# PROTEIN SEQUENCE + STRUCTURE CLASSIFICATION
# USING TEDBench-CATH
#
# COMPLETE SINGLE-FILE PIPELINE
#
# Pipeline:
#
# TEDBench/CATH
#      |
#      +--> Sequence preprocessing
#      |       |
#      |       +--> Amino acid composition
#      |       +--> Physicochemical features
#      |       +--> Dipeptide features
#      |
#      +--> 3D structure
#              |
#              +--> C-alpha geometry
#              +--> Radius of gyration
#              +--> End-to-end distance
#              +--> Contact density
#              +--> Contact order
#              +--> Backbone geometry
#              +--> Coordinate distribution
#
#              |
#              v
#       Feature Fusion
#              |
#              v
#      Random Forest Models
#              |
#      +-------+-------+
#      |       |       |
#      v       v       v
#    Sequence Structure Fusion
#
#              |
#              v
#       CATH Classification
#
# HPC:
#
# Sequential CPU
#       |
#       v
# Multiprocessing CPU
#       |
#       v
# Speedup / Efficiency
#
# ============================================================


# ============================================================
# 0. REQUIRED INSTALLATION
# ============================================================
#
# Run:
#
# pip install datasets pandas numpy scikit-learn
# pip install matplotlib seaborn joblib tqdm
#
# IMPORTANT:
# Numba is NOT required.
#
# ============================================================


import os
import time
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd

from collections import Counter

from tqdm import tqdm

from datasets import load_dataset

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

import joblib


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_NAME = "TEDBench/cath"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ------------------------------------------------------------
# FIRST RUN:
#
# Use 3000 or 5000
#
# AFTER EVERYTHING WORKS:
#
# MAX_PROTEINS = None
# ------------------------------------------------------------

MAX_PROTEINS = 3000


# Number of trees

N_ESTIMATORS = 300


# HPC benchmark size

BENCHMARK_PROTEINS = 500


# ------------------------------------------------------------
# Output directories
# ------------------------------------------------------------

OUTPUT_DIR = "protein_results"

FEATURE_DIR = os.path.join(
    OUTPUT_DIR,
    "features"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)


os.makedirs(
    FEATURE_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# ============================================================
# 2. AMINO ACID DEFINITIONS
# ============================================================

AMINO_ACIDS = list(
    "ACDEFGHIKLMNPQRSTVWY"
)


HYDROPHOBIC = set(
    "AFGILMPVWY"
)


CHARGED = set(
    "DEHKR"
)


POLAR = set(
    "CNQSTY"
)


AROMATIC = set(
    "FHWY"
)


# ============================================================
# 3. SEQUENCE CLEANING
# ============================================================

def clean_sequence(sequence):

    """
    Convert sequence to uppercase and retain
    only standard amino acids.
    """

    if sequence is None:

        return ""

    sequence = str(
        sequence
    ).upper()

    sequence = "".join(
        aa
        for aa in sequence
        if aa in AMINO_ACIDS
    )

    return sequence


# ============================================================
# 4. SEQUENCE FEATURE EXTRACTION
# ============================================================

def extract_basic_sequence_features(
    sequence
):

    """
    Extracts:

    20 amino acid composition features

    5 physicochemical features
    """

    length = len(
        sequence
    )

    features = {}


    # --------------------------------------------------------
    # Empty sequence
    # --------------------------------------------------------

    if length == 0:

        for aa in AMINO_ACIDS:

            features[
                f"{aa}_freq"
            ] = 0.0

        features[
            "sequence_length"
        ] = 0

        features[
            "hydrophobic_pct"
        ] = 0.0

        features[
            "charged_pct"
        ] = 0.0

        features[
            "polar_pct"
        ] = 0.0

        features[
            "aromatic_pct"
        ] = 0.0

        return features


    counts = Counter(
        sequence
    )


    # --------------------------------------------------------
    # Amino acid composition
    # --------------------------------------------------------

    for aa in AMINO_ACIDS:

        features[
            f"{aa}_freq"
        ] = (
            counts[aa]
            /
            length
        )


    # --------------------------------------------------------
    # Sequence length
    # --------------------------------------------------------

    features[
        "sequence_length"
    ] = length


    # --------------------------------------------------------
    # Hydrophobic percentage
    # --------------------------------------------------------

    features[
        "hydrophobic_pct"
    ] = (

        sum(
            aa in HYDROPHOBIC
            for aa in sequence
        )

        /
        length
    )


    # --------------------------------------------------------
    # Charged percentage
    # --------------------------------------------------------

    features[
        "charged_pct"
    ] = (

        sum(
            aa in CHARGED
            for aa in sequence
        )

        /
        length
    )


    # --------------------------------------------------------
    # Polar percentage
    # --------------------------------------------------------

    features[
        "polar_pct"
    ] = (

        sum(
            aa in POLAR
            for aa in sequence
        )

        /
        length
    )


    # --------------------------------------------------------
    # Aromatic percentage
    # --------------------------------------------------------

    features[
        "aromatic_pct"
    ] = (

        sum(
            aa in AROMATIC
            for aa in sequence
        )

        /
        length
    )


    return features


# ============================================================
# 5. FIND TOP DIPEPTIDES
# ============================================================

def find_top_dipeptides(
    dataset,
    top_n=50
):

    """
    Find the most frequently occurring
    dipeptides in the dataset.
    """

    counter = Counter()


    print(
        "\nFinding top dipeptides..."
    )


    for sample in tqdm(
        dataset,
        desc="Scanning sequences"
    ):

        sequence = clean_sequence(
            sample["sequence"]
        )


        for i in range(
            len(sequence) - 1
        ):

            dipeptide = (
                sequence[
                    i:i + 2
                ]
            )


            counter[
                dipeptide
            ] += 1


    top = counter.most_common(
        top_n
    )


    return [
        dp
        for dp, count in top
    ]


# ============================================================
# 6. DIPEPTIDE FEATURES
# ============================================================

def extract_dipeptide_features(
    sequence,
    top_dipeptides
):

    """
    Calculate normalized frequency
    for selected dipeptides.
    """

    features = {}

    length = len(
        sequence
    )


    if length < 2:

        for dp in top_dipeptides:

            features[
                f"di_{dp}"
            ] = 0.0

        return features


    counter = Counter()


    for i in range(
        length - 1
    ):

        dp = sequence[
            i:i + 2
        ]


        counter[
            dp
        ] += 1


    denominator = (
        length - 1
    )


    for dp in top_dipeptides:

        features[
            f"di_{dp}"
        ] = (

            counter[dp]
            /
            denominator
        )


    return features


# ============================================================
# 7. COMPLETE SEQUENCE FEATURE EXTRACTION
# ============================================================

def extract_sequence_features(
    dataset,
    top_dipeptides
):

    rows = []


    print("\n")
    print("=" * 70)
    print("SEQUENCE FEATURE EXTRACTION")
    print("=" * 70)


    for sample in tqdm(
        dataset,
        desc="Extracting sequence features"
    ):

        sequence = clean_sequence(
            sample["sequence"]
        )


        features = {}


        # 20 AA + 5 physicochemical

        features.update(
            extract_basic_sequence_features(
                sequence
            )
        )


        # 50 dipeptides

        features.update(
            extract_dipeptide_features(
                sequence,
                top_dipeptides
            )
        )


        # Metadata

        features[
            "name"
        ] = sample["name"]


        features[
            "label"
        ] = sample["label"]


        rows.append(
            features
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# 8. STRUCTURAL FEATURE EXTRACTION
# ============================================================

def get_ca_coordinates(
    coords
):

    """
    TEDBench coordinates are expected to be:

        [number_of_residues, 3, 3]

    where:

        0 = N
        1 = CA
        2 = C

    Therefore:

        coords[:, 1, :]

    gives C-alpha coordinates.
    """

    coords = np.asarray(
        coords,
        dtype=np.float32
    )


    if coords.ndim != 3:

        raise ValueError(
            "Unexpected coordinate shape: "
            + str(coords.shape)
        )


    if coords.shape[1] < 2:

        raise ValueError(
            "C-alpha coordinates not available."
        )


    ca = coords[
        :,
        1,
        :
    ]


    return ca


# ============================================================
# 9. RADIUS OF GYRATION
# ============================================================

def calculate_radius_gyration(
    ca
):

    """
    Measures how compact the protein structure is.
    """

    if len(ca) == 0:

        return 0.0


    center = np.mean(
        ca,
        axis=0
    )


    squared_distances = np.sum(

        (
            ca
            -
            center
        )
        ** 2,

        axis=1
    )


    return float(
        np.sqrt(
            np.mean(
                squared_distances
            )
        )
    )


# ============================================================
# 10. END-TO-END DISTANCE
# ============================================================

def calculate_end_to_end_distance(
    ca
):

    """
    Distance between first and last C-alpha.
    """

    if len(ca) < 2:

        return 0.0


    return float(
        np.linalg.norm(
            ca[-1]
            -
            ca[0]
        )
    )


# ============================================================
# 11. CONSECUTIVE C-ALPHA DISTANCES
# ============================================================

def calculate_ca_distances(
    ca
):

    if len(ca) < 2:

        return np.array([])


    return np.linalg.norm(

        ca[1:]
        -
        ca[:-1],

        axis=1
    )


# ============================================================
# 12. CONTACT FEATURES
# ============================================================

def calculate_contact_features(
    ca,
    threshold=8.0
):

    """
    Calculate structural contact features.

    Two residues are considered to have
    a spatial contact when their C-alpha
    distance is less than 8 Angstrom.

    Features:

        contact_density
        contact_order
        num_contacts
    """

    n = len(ca)


    if n < 3:

        return {

            "contact_density":
                0.0,

            "contact_order":
                0.0,

            "num_contacts":
                0
        }


    # --------------------------------------------------------
    # Pairwise distance matrix
    # --------------------------------------------------------

    distance_matrix = np.linalg.norm(

        ca[:, None, :]
        -
        ca[None, :, :],

        axis=-1
    )


    # --------------------------------------------------------
    # Contact matrix
    # --------------------------------------------------------

    contact_matrix = (

        distance_matrix
        <
        threshold
    )


    # Remove self-contact

    np.fill_diagonal(
        contact_matrix,
        False
    )


    # --------------------------------------------------------
    # Remove adjacent residues
    # --------------------------------------------------------

    for i in range(
        n - 1
    ):

        contact_matrix[
            i,
            i + 1
        ] = False


        contact_matrix[
            i + 1,
            i
        ] = False


    # --------------------------------------------------------
    # Upper triangular matrix
    # --------------------------------------------------------

    upper = np.triu(
        contact_matrix,
        k=1
    )


    contact_indices = np.argwhere(
        upper
    )


    num_contacts = len(
        contact_indices
    )


    # --------------------------------------------------------
    # Contact density
    # --------------------------------------------------------

    possible_contacts = (
        n * (n - 1) / 2
    )


    contact_density = (

        num_contacts
        /
        possible_contacts
    )


    # --------------------------------------------------------
    # Contact order
    # --------------------------------------------------------

    if num_contacts > 0:

        sequence_separations = (

            np.abs(

                contact_indices[:, 0]
                -
                contact_indices[:, 1]
            )
        )


        contact_order = (

            np.mean(
                sequence_separations
            )
            /
            n
        )

    else:

        contact_order = 0.0


    return {

        "contact_density":
            float(
                contact_density
            ),

        "contact_order":
            float(
                contact_order
            ),

        "num_contacts":
            int(
                num_contacts
            )
    }


# ============================================================
# 13. BACKBONE GEOMETRY
# ============================================================

def calculate_backbone_geometry(
    coords
):

    """
    Calculate basic backbone distances:

        N - CA
        CA - C
        C(i) - N(i+1)
    """

    coords = np.asarray(
        coords,
        dtype=np.float32
    )


    if len(coords) == 0:

        return {

            "mean_N_CA":
                np.nan,

            "std_N_CA":
                np.nan,

            "mean_CA_C":
                np.nan,

            "std_CA_C":
                np.nan,

            "mean_C_N":
                np.nan,

            "std_C_N":
                np.nan
        }


    N = coords[
        :,
        0,
        :
    ]


    CA = coords[
        :,
        1,
        :
    ]


    C = coords[
        :,
        2,
        :
    ]


    # N - CA

    N_CA = np.linalg.norm(

        N
        -
        CA,

        axis=1
    )


    # CA - C

    CA_C = np.linalg.norm(

        CA
        -
        C,

        axis=1
    )


    # C(i) - N(i+1)

    if len(coords) > 1:

        C_N = np.linalg.norm(

            C[:-1]
            -
            N[1:],

            axis=1
        )

    else:

        C_N = np.array([])


    return {

        "mean_N_CA":
            float(
                np.mean(
                    N_CA
                )
            ),

        "std_N_CA":
            float(
                np.std(
                    N_CA
                )
            ),

        "mean_CA_C":
            float(
                np.mean(
                    CA_C
                )
            ),

        "std_CA_C":
            float(
                np.std(
                    CA_C
                )
            ),

        "mean_C_N":
            (
                float(
                    np.mean(
                        C_N
                    )
                )
                if len(C_N) > 0
                else np.nan
            ),

        "std_C_N":
            (
                float(
                    np.std(
                        C_N
                    )
                )
                if len(C_N) > 0
                else np.nan
            )
    }


# ============================================================
# 14. STRUCTURAL FEATURE EXTRACTION
# ============================================================

def extract_structural_features(
    sample
):

    """
    Extract numerical descriptors from
    the experimentally supplied 3D structure.
    """

    coords = np.asarray(
        sample["coords"],
        dtype=np.float32
    )


    ca = get_ca_coordinates(
        coords
    )


    features = {}


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    features[
        "name"
    ] = sample["name"]


    features[
        "label"
    ] = sample["label"]


    features[
        "structure_length"
    ] = len(ca)


    # --------------------------------------------------------
    # Radius of gyration
    # --------------------------------------------------------

    features[
        "radius_gyration"
    ] = calculate_radius_gyration(
        ca
    )


    # --------------------------------------------------------
    # End-to-end distance
    # --------------------------------------------------------

    features[
        "end_to_end_distance"
    ] = calculate_end_to_end_distance(
        ca
    )


    # --------------------------------------------------------
    # C-alpha distances
    # --------------------------------------------------------

    ca_distances = (
        calculate_ca_distances(
            ca
        )
    )


    if len(ca_distances) > 0:

        features[
            "mean_ca_distance"
        ] = float(
            np.mean(
                ca_distances
            )
        )


        features[
            "std_ca_distance"
        ] = float(
            np.std(
                ca_distances
            )
        )


        features[
            "min_ca_distance"
        ] = float(
            np.min(
                ca_distances
            )
        )


        features[
            "max_ca_distance"
        ] = float(
            np.max(
                ca_distances
            )
        )

    else:

        features[
            "mean_ca_distance"
        ] = np.nan

        features[
            "std_ca_distance"
        ] = np.nan

        features[
            "min_ca_distance"
        ] = np.nan

        features[
            "max_ca_distance"
        ] = np.nan


    # --------------------------------------------------------
    # Contact features
    # --------------------------------------------------------

    contact_features = (
        calculate_contact_features(
            ca
        )
    )


    features.update(
        contact_features
    )


    # --------------------------------------------------------
    # Backbone features
    # --------------------------------------------------------

    backbone_features = (
        calculate_backbone_geometry(
            coords
        )
    )


    features.update(
        backbone_features
    )


    # --------------------------------------------------------
    # Spatial coordinate distribution
    # --------------------------------------------------------

    if len(ca) > 0:

        features[
            "x_std"
        ] = float(
            np.std(
                ca[:, 0]
            )
        )


        features[
            "y_std"
        ] = float(
            np.std(
                ca[:, 1]
            )
        )


        features[
            "z_std"
        ] = float(
            np.std(
                ca[:, 2]
            )
        )

    else:

        features[
            "x_std"
        ] = np.nan

        features[
            "y_std"
        ] = np.nan

        features[
            "z_std"
        ] = np.nan


    return features


# ============================================================
# 15. STRUCTURAL FEATURE EXTRACTION
# ============================================================

def extract_all_structural_features(
    dataset
):

    rows = []


    print("\n")
    print("=" * 70)
    print("STRUCTURAL FEATURE EXTRACTION")
    print("=" * 70)


    for sample in tqdm(
        dataset,
        desc="Extracting structural features"
    ):

        try:

            features = (
                extract_structural_features(
                    sample
                )
            )


            rows.append(
                features
            )


        except Exception as e:

            print(
                "\nError processing:",
                sample["name"]
            )


            print(
                "Error:",
                e
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. SEQUENTIAL HPC FUNCTION
# ============================================================

def sequential_radius_gyration(
    ca
):

    """
    Intentionally implemented using
    explicit loops for the HPC benchmark.

    This gives us a computational workload
    that can later be implemented using:

        Sequential C/C++
        OpenMP
        CUDA
    """

    n = len(ca)


    if n == 0:

        return 0.0


    # --------------------------------------------------------
    # Calculate centroid
    # --------------------------------------------------------

    cx = 0.0
    cy = 0.0
    cz = 0.0


    for i in range(n):

        cx += ca[i][0]

        cy += ca[i][1]

        cz += ca[i][2]


    cx /= n
    cy /= n
    cz /= n


    # --------------------------------------------------------
    # Calculate squared distances
    # --------------------------------------------------------

    total = 0.0


    for i in range(n):

        dx = (
            ca[i][0]
            -
            cx
        )


        dy = (
            ca[i][1]
            -
            cy
        )


        dz = (
            ca[i][2]
            -
            cz
        )


        total += (

            dx * dx
            +
            dy * dy
            +
            dz * dz
        )


    return np.sqrt(
        total / n
    )


# ============================================================
# 17. PARALLEL WORKER
# ============================================================

def parallel_worker(
    coords
):

    """
    Worker function executed by
    separate CPU processes.
    """

    coords = np.asarray(
        coords,
        dtype=np.float32
    )


    ca = coords[
        :,
        1,
        :
    ]


    return sequential_radius_gyration(
        ca
    )


# ============================================================
# 18. HPC BENCHMARK
# ============================================================

def run_hpc_benchmark(
    dataset
):

    print("\n")
    print("=" * 70)
    print("HPC STRUCTURAL FEATURE BENCHMARK")
    print("=" * 70)


    benchmark_count = min(

        BENCHMARK_PROTEINS,

        len(dataset)
    )


    print(
        "\nBenchmarking",
        benchmark_count,
        "proteins..."
    )


    # --------------------------------------------------------
    # Prepare coordinates
    # --------------------------------------------------------

    benchmark_data = []


    for i in range(
        benchmark_count
    ):

        benchmark_data.append(
            dataset[i]["coords"]
        )


    # ========================================================
    # SEQUENTIAL
    # ========================================================

    print(
        "\nRunning sequential calculation..."
    )


    sequential_results = []


    start = time.perf_counter()


    for coords in tqdm(
        benchmark_data,
        desc="Sequential"
    ):

        coords = np.asarray(
            coords,
            dtype=np.float32
        )


        ca = coords[
            :,
            1,
            :
        ]


        result = (
            sequential_radius_gyration(
                ca
            )
        )


        sequential_results.append(
            result
        )


    sequential_time = (
        time.perf_counter()
        -
        start
    )


    print(
        "\nSequential time:",
        round(
            sequential_time,
            4
        ),
        "seconds"
    )


    # ========================================================
    # PARALLEL CPU
    # ========================================================

    # --------------------------------------------------------
    # Windows-safe multiprocessing
    # --------------------------------------------------------

    cpu_count = (
        os.cpu_count()
        or 1
    )


    # Leave one core free

    workers = max(
        1,
        cpu_count - 1
    )


    print(
        "\nAvailable CPU cores:",
        cpu_count
    )


    print(
        "Parallel workers:",
        workers
    )


    print(
        "\nRunning parallel CPU calculation..."
    )


    start = time.perf_counter()


    with mp.Pool(
        processes=workers
    ) as pool:

        parallel_results = list(

            tqdm(

                pool.imap(
                    parallel_worker,
                    benchmark_data
                ),

                total=benchmark_count,

                desc="Parallel"
            )
        )


    parallel_time = (
        time.perf_counter()
        -
        start
    )


    print(
        "\nParallel time:",
        round(
            parallel_time,
            4
        ),
        "seconds"
    )


    # ========================================================
    # SPEEDUP
    # ========================================================

    speedup = (

        sequential_time
        /
        parallel_time
    )


    # ========================================================
    # EFFICIENCY
    # ========================================================

    efficiency = (

        speedup
        /
        workers
    )


    # ========================================================
    # NUMERICAL VALIDATION
    # ========================================================

    max_difference = np.max(

        np.abs(

            np.array(
                sequential_results
            )

            -

            np.array(
                parallel_results
            )
        )
    )


    print("\n")
    print("=" * 70)
    print("HPC RESULTS")
    print("=" * 70)


    print(
        "\nSequential time:",
        round(
            sequential_time,
            4
        ),
        "seconds"
    )


    print(
        "Parallel time:",
        round(
            parallel_time,
            4
        ),
        "seconds"
    )


    print(
        "Number of workers:",
        workers
    )


    print(
        "Speedup:",
        round(
            speedup,
            4
        ),
        "x"
    )


    print(
        "Parallel efficiency:",
        round(
            efficiency * 100,
            2
        ),
        "%"
    )


    print(
        "Maximum numerical difference:",
        max_difference
    )


    if max_difference < 1e-5:

        print(
            "\n✓ Sequential and parallel results match."
        )

    else:

        print(
            "\n⚠ Numerical difference detected."
        )


    results = pd.DataFrame({

        "Metric": [

            "Sequential Time",

            "Parallel Time",

            "CPU Workers",

            "Speedup",

            "Efficiency (%)",

            "Maximum Numerical Difference"
        ],

        "Value": [

            sequential_time,

            parallel_time,

            workers,

            speedup,

            efficiency * 100,

            max_difference
        ]
    })


    results.to_csv(

        os.path.join(

            OUTPUT_DIR,

            "hpc_benchmark.csv"
        ),

        index=False
    )


    return results


# ============================================================
# 19. MODEL TRAINING
# ============================================================

def train_random_forest(
    X_train,
    y_train
):

    model = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        class_weight="balanced",

        n_jobs=-1,

        random_state=RANDOM_STATE,

        min_samples_leaf=2,

        max_features="sqrt"
    )


    model.fit(
        X_train,
        y_train
    )


    return model


# ============================================================
# 20. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    predictions = (
        model.predict(
            X_test
        )
    )


    accuracy = (
        accuracy_score(
            y_test,
            predictions
        )
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )


    macro_f1 = (
        f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
    )


    weighted_f1 = (
        f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    )


    print("\n")
    print("=" * 70)
    print(
        model_name
    )
    print("=" * 70)


    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )


    print(
        "Balanced Accuracy:",
        round(
            balanced_accuracy,
            4
        )
    )


    print(
        "Macro F1:",
        round(
            macro_f1,
            4
        )
    )


    print(
        "Weighted F1:",
        round(
            weighted_f1,
            4
        )
    )


    return {

        "Model":
            model_name,

        "Accuracy":
            accuracy,

        "Balanced_Accuracy":
            balanced_accuracy,

        "Macro_F1":
            macro_f1,

        "Weighted_F1":
            weighted_f1,

        "Predictions":
            predictions
    }


# ============================================================
# 21. MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("PROTEIN FAMILY / STRUCTURE CLASSIFICATION PIPELINE")
    print("=" * 70)


    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LOADING TEDBENCH-CATH")
    print("=" * 70)


    dataset = load_dataset(

        DATASET_NAME,

        split="test"
    )


    print(
        "\nDataset loaded successfully."
    )


    print(
        "Total proteins:",
        len(dataset)
    )


    print(
        "\nDataset columns:"
    )


    print(
        dataset.column_names
    )


    # --------------------------------------------------------
    # Dataset size
    # --------------------------------------------------------

    if MAX_PROTEINS is not None:

        n = min(

            MAX_PROTEINS,

            len(dataset)
        )


        dataset = dataset.select(
            range(n)
        )


        print(
            "\nUsing",
            n,
            "proteins for this run."
        )

    else:

        print(
            "\nUsing complete dataset."
        )


    # ========================================================
    # LABEL INFORMATION
    # ========================================================

    label_feature = dataset.features[
        "label"
    ]


    print(
        "\nNumber of CATH classes:",
        label_feature.num_classes
    )


    print(
        "\nExample labels:"
    )


    for i in range(

        min(
            10,
            label_feature.num_classes
        )
    ):

        print(

            i,
            "->",

            label_feature.int2str(i)
        )


    # ========================================================
    # SEQUENCE FEATURES
    # ========================================================

    top_dipeptides = (
        find_top_dipeptides(
            dataset,
            top_n=50
        )
    )


    print(
        "\nTop 50 dipeptides:"
    )


    print(
        top_dipeptides
    )


    sequence_df = (
        extract_sequence_features(

            dataset,

            top_dipeptides
        )
    )


    print(
        "\nSequence feature matrix:",
        sequence_df.shape
    )


    sequence_df.to_csv(

        os.path.join(

            FEATURE_DIR,

            "sequence_features.csv"
        ),

        index=False
    )


    # ========================================================
    # STRUCTURAL FEATURES
    # ========================================================

    structural_df = (
        extract_all_structural_features(
            dataset
        )
    )


    print(
        "\nStructural feature matrix:",
        structural_df.shape
    )


    structural_df.to_csv(

        os.path.join(

            FEATURE_DIR,

            "structural_features.csv"
        ),

        index=False
    )


    # ========================================================
    # FEATURE FUSION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FEATURE FUSION")
    print("=" * 70)


    df = pd.merge(

        sequence_df,

        structural_df,

        on=[
            "name",
            "label"
        ],

        how="inner"
    )


    print(
        "\nMerged dataset:",
        df.shape
    )


    # ========================================================
    # FEATURE LISTS
    # ========================================================

    sequence_features = [

        col

        for col in sequence_df.columns

        if col not in [
            "name",
            "label"
        ]
    ]


    structural_features = [

        col

        for col in structural_df.columns

        if col not in [
            "name",
            "label"
        ]
    ]


    fusion_features = (

        sequence_features
        +
        structural_features
    )


    print(
        "\nSequence features:",
        len(sequence_features)
    )


    print(
        "Structural features:",
        len(structural_features)
    )


    print(
        "Total fused features:",
        len(fusion_features)
    )


    # ========================================================
    # TARGET
    # ========================================================

    y = df[
        "label"
    ].astype(int)


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)


    class_counts = (
        y.value_counts()
    )


    print(
        "\nClasses represented:",
        len(class_counts)
    )


    print(
        "\nTop 10 classes:"
    )


    print(
        class_counts.head(10)
    )


    # ========================================================
    # FEATURE MATRICES
    # ========================================================

    X_sequence = df[
        sequence_features
    ].copy()


    X_structure = df[
        structural_features
    ].copy()


    X_fusion = df[
        fusion_features
    ].copy()


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    indices = np.arange(
        len(df)
    )


    try:

        train_idx, test_idx = (

            train_test_split(

                indices,

                test_size=TEST_SIZE,

                random_state=RANDOM_STATE,

                stratify=y
            )
        )

    except ValueError:

        print(
            "\nWarning: stratified split failed."
        )


        print(
            "Using normal random split."
        )


        train_idx, test_idx = (

            train_test_split(

                indices,

                test_size=TEST_SIZE,

                random_state=RANDOM_STATE
            )
        )


    # ========================================================
    # IMPUTATION
    # ========================================================

    sequence_imputer = (
        SimpleImputer(
            strategy="median"
        )
    )


    structure_imputer = (
        SimpleImputer(
            strategy="median"
        )
    )


    fusion_imputer = (
        SimpleImputer(
            strategy="median"
        )
    )


    X_sequence = (
        sequence_imputer.fit_transform(
            X_sequence
        )
    )


    X_structure = (
        structure_imputer.fit_transform(
            X_structure
        )
    )


    X_fusion = (
        fusion_imputer.fit_transform(
            X_fusion
        )
    )


    # ========================================================
    # SPLIT MATRICES
    # ========================================================

    X_seq_train = X_sequence[
        train_idx
    ]


    X_seq_test = X_sequence[
        test_idx
    ]


    X_struct_train = X_structure[
        train_idx
    ]


    X_struct_test = X_structure[
        test_idx
    ]


    X_fusion_train = X_fusion[
        train_idx
    ]


    X_fusion_test = X_fusion[
        test_idx
    ]


    y_train = y.iloc[
        train_idx
    ]


    y_test = y.iloc[
        test_idx
    ]


    print(
        "\nTraining samples:",
        len(train_idx)
    )


    print(
        "Testing samples:",
        len(test_idx)
    )


    # ========================================================
    # SEQUENCE MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MODEL 1: SEQUENCE ONLY")
    print("=" * 70)


    start = time.perf_counter()


    sequence_model = (
        train_random_forest(

            X_seq_train,

            y_train
        )
    )


    sequence_training_time = (

        time.perf_counter()
        -
        start
    )


    sequence_result = (
        evaluate_model(

            sequence_model,

            X_seq_test,

            y_test,

            "Sequence Only"
        )
    )


    print(
        "Training time:",
        round(
            sequence_training_time,
            4
        ),
        "seconds"
    )


    # ========================================================
    # STRUCTURE MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MODEL 2: STRUCTURE ONLY")
    print("=" * 70)


    start = time.perf_counter()


    structure_model = (
        train_random_forest(

            X_struct_train,

            y_train
        )
    )


    structure_training_time = (

        time.perf_counter()
        -
        start
    )


    structure_result = (
        evaluate_model(

            structure_model,

            X_struct_test,

            y_test,

            "Structure Only"
        )
    )


    print(
        "Training time:",
        round(
            structure_training_time,
            4
        ),
        "seconds"
    )


    # ========================================================
    # FUSION MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MODEL 3: SEQUENCE + STRUCTURE")
    print("=" * 70)


    start = time.perf_counter()


    fusion_model = (
        train_random_forest(

            X_fusion_train,

            y_train
        )
    )


    fusion_training_time = (

        time.perf_counter()
        -
        start
    )


    fusion_result = (
        evaluate_model(

            fusion_model,

            X_fusion_test,

            y_test,

            "Sequence + Structure"
        )
    )


    print(
        "Training time:",
        round(
            fusion_training_time,
            4
        ),
        "seconds"
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame([

        {
            key: value

            for key, value
            in sequence_result.items()

            if key != "Predictions"
        },

        {
            key: value

            for key, value
            in structure_result.items()

            if key != "Predictions"
        },

        {
            key: value

            for key, value
            in fusion_result.items()

            if key != "Predictions"
        }
    ])


    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)


    print(
        comparison_df.to_string(
            index=False
        )
    )


    comparison_df.to_csv(

        os.path.join(

            OUTPUT_DIR,

            "model_comparison.csv"
        ),

        index=False
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)


    importance_df = pd.DataFrame({

        "feature":
            fusion_features,

        "importance":
            fusion_model.feature_importances_
    })


    importance_df = (

        importance_df

        .sort_values(

            "importance",

            ascending=False
        )
    )


    print(
        "\nTop 20 features:"
    )


    print(
        importance_df.head(20).to_string(
            index=False
        )
    )


    importance_df.to_csv(

        os.path.join(

            OUTPUT_DIR,

            "feature_importance.csv"
        ),

        index=False
    )


    # ========================================================
    # FEATURE IMPORTANCE PLOT
    # ========================================================

    top_features = (

        importance_df

        .head(20)

        .sort_values(
            "importance"
        )
    )


    plt.figure(
        figsize=(10, 8)
    )


    plt.barh(

        top_features[
            "feature"
        ],

        top_features[
            "importance"
        ]
    )


    plt.xlabel(
        "Random Forest Importance"
    )


    plt.ylabel(
        "Feature"
    )


    plt.title(
        "Top 20 Features - Sequence + Structure"
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "feature_importance.png"
        ),

        dpi=300
    )


    plt.close()


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    fusion_predictions = (
        fusion_result[
            "Predictions"
        ]
    )


    cm = confusion_matrix(

        y_test,

        fusion_predictions
    )


    plt.figure(
        figsize=(14, 12)
    )


    sns.heatmap(
        cm,
        cmap="Blues",
        cbar=True
    )


    plt.xlabel(
        "Predicted Class"
    )


    plt.ylabel(
        "True Class"
    )


    plt.title(
        "Confusion Matrix - Sequence + Structure"
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "confusion_matrix.png"
        ),

        dpi=300
    )


    plt.close()


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        y_test,

        fusion_predictions,

        zero_division=0
    )


    print("\n")
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)


    print(
        report
    )


    with open(

        os.path.join(

            OUTPUT_DIR,

            "classification_report.txt"
        ),

        "w"
    ) as f:

        f.write(
            report
        )


    # ========================================================
    # SAVE MODELS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SAVING MODELS")
    print("=" * 70)


    joblib.dump(

        sequence_model,

        os.path.join(

            MODEL_DIR,

            "sequence_model.pkl"
        )
    )


    joblib.dump(

        structure_model,

        os.path.join(

            MODEL_DIR,

            "structure_model.pkl"
        )
    )


    joblib.dump(

        fusion_model,

        os.path.join(

            MODEL_DIR,

            "fusion_model.pkl"
        )
    )


    joblib.dump(

        sequence_imputer,

        os.path.join(

            MODEL_DIR,

            "sequence_imputer.pkl"
        )
    )


    joblib.dump(

        structure_imputer,

        os.path.join(

            MODEL_DIR,

            "structure_imputer.pkl"
        )
    )


    joblib.dump(

        fusion_imputer,

        os.path.join(

            MODEL_DIR,

            "fusion_imputer.pkl"
        )
    )


    joblib.dump(

        sequence_features,

        os.path.join(

            MODEL_DIR,

            "sequence_feature_names.pkl"
        )
    )


    joblib.dump(

        structural_features,

        os.path.join(

            MODEL_DIR,

            "structural_feature_names.pkl"
        )
    )


    # ========================================================
    # EXAMPLE PREDICTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EXAMPLE PROTEIN PREDICTION")
    print("=" * 70)


    example_index = 0


    example_name = df[
        "name"
    ].iloc[
        example_index
    ]


    example_true_label = df[
        "label"
    ].iloc[
        example_index
    ]


    example_seq = X_sequence[
        example_index
    ].reshape(
        1,
        -1
    )


    example_struct = X_structure[
        example_index
    ].reshape(
        1,
        -1
    )


    example_fusion = X_fusion[
        example_index
    ].reshape(
        1,
        -1
    )


    seq_prediction = (
        sequence_model.predict(
            example_seq
        )[0]
    )


    struct_prediction = (
        structure_model.predict(
            example_struct
        )[0]
    )


    fusion_prediction = (
        fusion_model.predict(
            example_fusion
        )[0]
    )


    try:

        true_cath = (
            label_feature.int2str(
                int(
                    example_true_label
                )
            )
        )


        seq_cath = (
            label_feature.int2str(
                int(
                    seq_prediction
                )
            )
        )


        struct_cath = (
            label_feature.int2str(
                int(
                    struct_prediction
                )
            )
        )


        fusion_cath = (
            label_feature.int2str(
                int(
                    fusion_prediction
                )
            )
        )


    except Exception:

        true_cath = str(
            example_true_label
        )

        seq_cath = str(
            seq_prediction
        )

        struct_cath = str(
            struct_prediction
        )

        fusion_cath = str(
            fusion_prediction
        )


    print(
        "\nProtein:",
        example_name
    )


    print(
        "\nTrue CATH label:",
        true_cath
    )


    print(
        "Sequence prediction:",
        seq_cath
    )


    print(
        "Structure prediction:",
        struct_cath
    )


    print(
        "Sequence + Structure prediction:",
        fusion_cath
    )


    # ========================================================
    # HPC BENCHMARK
    # ========================================================

    hpc_results = run_hpc_benchmark(
        dataset
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


    print(
        "\nDataset:",
        DATASET_NAME
    )


    print(
        "Proteins processed:",
        len(dataset)
    )


    print(
        "Sequence features:",
        len(sequence_features)
    )


    print(
        "Structural features:",
        len(structural_features)
    )


    print(
        "Fused features:",
        len(fusion_features)
    )


    print(
        "\nModels:"
    )


    print(
        "1. Sequence-only Random Forest"
    )


    print(
        "2. Structure-only Random Forest"
    )


    print(
        "3. Sequence + Structure Random Forest"
    )


    print(
        "\nHPC:"
    )


    print(
        "1. Sequential CPU"
    )


    print(
        "2. Parallel CPU"
    )


    print(
        "\nResults saved to:"
    )


    print(
        OUTPUT_DIR
    )


    print(
        "\nImportant files:"
    )


    print(
        "  features/sequence_features.csv"
    )


    print(
        "  features/structural_features.csv"
    )


    print(
        "  model_comparison.csv"
    )


    print(
        "  feature_importance.csv"
    )


    print(
        "  classification_report.txt"
    )


    print(
        "  hpc_benchmark.csv"
    )


    print(
        "  plots/feature_importance.png"
    )


    print(
        "  plots/confusion_matrix.png"
    )


    print(
        "  models/*.pkl"
    )


    print("\nDone!")


# ============================================================
# 22. WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()