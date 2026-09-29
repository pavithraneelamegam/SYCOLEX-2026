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
## Overview

The statute prediction pipeline represents IPC sections as **textual prototypes** using their section descriptions. Case facts are segmented into sentence-level units and encoded with the same InLegalBERT encoder used for the statute prototypes.

The system consists of two prototype-based approaches:

- **Run 1 -- Prototype-Contrastive InLegalBERT**
- **Run 2 -- Frozen InLegalBERT Prototype Baseline**

The prototype bank is loaded from `ipc_sections_clean.json`. In the provided runs, the prototype catalog contains **574 IPC statute prototypes**. The supervised dataset contains **525 cases**, with **7 IPC sections** observed as gold labels.

### Dataset Split

The cases are divided into:

- **70% training**
- **10% validation**
- **20% testing**

### Prototype-Based Statute Prediction

**Run 1 -- Prototype-Contrastive InLegalBERT:** InLegalBERT is contrastively fine-tuned using an InfoNCE objective to align case sentences with IPC statute prototypes.

**Run 2 -- Frozen InLegalBERT:** Uses the original InLegalBERT representations for prototype-based statute matching, followed by per-class logistic regression.

Both runs use **575 IPC statute prototypes** and predict among **7 supervised IPC sections**. Supporting evidence is retrieved using **BM25 (0.30), cosine similarity (0.40), and classifier relevance (0.30)**, with the **top 3 evidence sentences** selected for each predicted statute.
