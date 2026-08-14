# HPC-Accelerated Protein Sequence Analysis and Functional Classification Using Machine Learning

### Objectives

* Preprocess and analyze protein sequences.
* Extract biologically meaningful features.
* Classify protein families using machine learning.
* Implement feature extraction using:

  * Sequential CPU
  * OpenMP CPU
  * CUDA GPU
* Compare execution time, speedup, efficiency, and scalability.
* Compare ML models based on classification performance.

## Workflow

```text
Protein Dataset
      ↓
Preprocessing
      ↓
Feature Extraction
      ↓
 ┌────┼──────────────┐
 ↓    ↓              ↓
AA   k-mers   Physicochemical
 ↓    ↓              ↓
 └────┼──────────────┘
      ↓
Feature Matrix
      ↓
ML Classification
      ↓
Random Forest / SVM / XGBoost
      ↓
Protein Family Prediction
      ↓
HPC Benchmarking
      ↓
Sequential CPU vs OpenMP vs CUDA
      ↓
Performance Analysis
```

## Dataset

Primary dataset: **Pfam Protein Family Classification Dataset**

* ~46,872 protein sequences
* 62 protein-family classes
* Suitable for multiclass classification

## Features

* Amino-acid composition
* k-mer frequencies
* Physicochemical properties

Optional advanced feature:

* ProtT5/ESM protein embeddings

## Machine Learning

Models:

* Random Forest
* SVM
* XGBoost

Metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## HPC

### Sequential CPU

Baseline implementation for feature extraction.

### OpenMP

Parallel feature extraction using multiple CPU threads.

### CUDA

GPU acceleration of computationally intensive feature extraction.

### Performance Metrics

```text
Speedup = Sequential Time / Parallel Time

Efficiency = Speedup / Number of Threads

Throughput = Sequences / Execution Time
```

## Experiments

The project compares:

* Different feature representations
* Different ML models
* Sequential vs OpenMP vs CUDA
* Different CPU thread counts
* Different dataset sizes

## Future Extensions 

* ProtT5/ESM embeddings
* Gene Ontology prediction
* CNN/LSTM/Transformer models
* MPI-based distributed processing
* Cloud GPU deployment
