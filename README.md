# SYCOLEX-2026

This repository contains the implementation for the **FIRE 2026 Shared Tasks** on:

1. **Explainable Statute Prediction (Task 1)**
2. **Sycophancy Detection (Task 2)**

The repository implements prototype-based statute prediction with **InLegalBERT** and a dual-branch **TRUE--FLIP** architecture for sycophancy detection.

## Repository Contents

| File | Description |
|---|---|
| `Contrastive Protypical Learning- Statute Prediction.ipynb` | **Run 1:** Prototype-based explainable statute prediction using contrastive fine-tuning of InLegalBERT |
| `InlegalBERT- Statute Prediction.ipynb` | **Run 2:** Prototype-based statute prediction using the original/frozen InLegalBERT representations |
| `sycophancy_dualbranch.ipynb` | Dual-branch TRUE--FLIP sycophancy detection model |
| `sycophancy_dualbranch_exclude Facts.ipynb` | No-facts baseline for TRUE--FLIP sycophancy detection |

## Overview

The statute prediction pipeline represents IPC sections as **textual prototypes** using their section descriptions. Case facts are segmented into sentence-level units and encoded using **[law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)**.

The system consists of two prototype-based approaches:

- **Run 1 -- Prototype-Contrastive InLegalBERT**
- **Run 2 -- Frozen InLegalBERT Prototype Baseline**

The IPC statute descriptions are loaded from a **Kaggle IPC sections dataset**, providing **574 IPC statute prototypes**. The supervised dataset contains **525 cases**, with **7 IPC sections** observed as gold labels.

### Dataset Split

The cases are divided into:

- **70% training**
- **10% validation**
- **20% testing**

### Prototype-Based Statute Prediction

**Run 1 -- Prototype-Contrastive InLegalBERT:** **[law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)** is contrastively fine-tuned using an InfoNCE objective to align case sentences with IPC statute prototypes.

**Run 2 -- Frozen InLegalBERT:** Uses the original **[law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)** representations for prototype-based statute matching, followed by per-class logistic regression.

Both runs use **574 IPC statute prototypes** and predict among **7 supervised IPC sections**. Supporting evidence is retrieved using **BM25 (0.30), cosine similarity (0.40), and classifier relevance (0.30)**, with the **top 3 evidence sentences** selected for each predicted statute. **[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)** is then used to generate explanations based on the predicted statute and retrieved supporting evidence.

# Task 2: Sycophancy Detection

## Problem Formulation

The sycophancy task is formulated as **binary classification** over TRUE--FLIP response pairs. The model receives a **TRUE response** to the original perspective and a **FLIP response** to the opposing perspective, and learns their relationship.

## Dual-Branch Architecture

The main model uses a **Dual-Branch Hierarchical [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)** architecture. TRUE and FLIP inputs are independently chunked and encoded, followed by attention pooling.

The resulting representations are combined using:

- TRUE embedding
- FLIP embedding
- Element-wise difference
- Element-wise product

These features are concatenated and passed to an **MLP classifier** for binary prediction.

## No-Facts Baseline

The repository also includes `sycophancy_dualbranch_exclude Facts.ipynb`, which removes case facts and uses only the TRUE and FLIP instructions and responses.

The experiment contains **10,620 examples**, split into **8,472 training** and **2,148 validation** examples using a group-based split by `case_id`. No common cases are present between the training and validation sets.
