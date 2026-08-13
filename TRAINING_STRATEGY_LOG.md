# Architectural Decisions & Training Strategy Log

This document serves as an official log of the advanced architectural design decisions and the Distributed MLOps training strategy formulated for the **Onco_Bot** final year project. It provides the definitive answers required to defend the architecture to academic panels.

## 1. Biological Constraining via KEGG Pathways
* **The Problem:** Training a neural network on 20,000 unconstrained genes using only 403 patients leads to catastrophic overfitting (The Curse of Dimensionality) and creates a "black-box" model.
* **The Solution:** We parsed the official 320 KEGG Human Pathways from the Broad Institute MSigDB into a `.gmt` file. This is injected as a PyTorch Boolean Mask into the `PathwayAwareTransformer`.
* **Clinical Impact:** The network is physically restricted to evaluating known biological systems. The Cross-Attention mechanism can specifically highlight which pathway (e.g., p53, ERBB, VEGF) is interacting with the tumor morphology, granting true Explainable AI (XAI).

## 2. Distributed MLOps Pipeline (Protecting Local Hardware)
Processing 4GB Whole-Slide Images (WSIs) dynamically during multimodal training will immediately bottleneck local consumer hardware (RAM crashes and SSD wear via OS swapping). 

We engineered a split-pipeline approach:

### Phase A: Cloud Feature Extraction (The "Generalist")
* **Platform:** Google Colab Pro (V100/A100 GPUs)
* **Model:** Off-the-shelf DINO Vision Transformer.
* **Process:** The heavy `.svs` images are chunked into 1,024 patches, pushed through DINO, and compressed into highly dense 384-dimensional mathematical vectors.
* **Output:** A tiny (~1.5MB) `.pt` file representing the morphological features of the tumor. *Note: These files are the data representations, not the trained network weights.*
* **Compute Cost:** Extracting features for the 403 WSI dataset takes approximately 3-4 hours on a V100 GPU, costing ~20 Compute Units (easily covered by the base Colab Pro subscription).

### Phase B: Local Edge Training (The "Specialist")
* **Platform:** Local Machine (Apple Silicon Mac)
* **Dataset:** `aligned_3way_slides_MINIMAL_case_level.csv` (N = 403 paired patients).
* **Process:** The pre-extracted `.pt` WSI tensors and the `.tsv` genomic data are fed into the custom `CrossAttentionFusion` model and the `CoxSurvivalHead`. 
* **Benefit:** By training the Multimodal Fusion network on pre-extracted embeddings rather than raw pixels, local training converges rapidly without ever touching the SSD swap memory.

## 3. Addressing the "Missing Modality" Problem
* **Current State:** The Cross-Attention mechanism strictly requires both modalities (Genomic = Query, Pathology = Key/Value) to execute the matrix multiplication. 
* **Real-World Deployment Strategy:** If a future clinical patient has biopsy data but no DNA sequencing, the standard architectural fallback is **Zero-Imputation**. A tensor of zeros is passed to the Genomic branch, and the model (if trained with modality dropout) shifts 100% of the prediction weight to the Vision branch.

## 4. Evaluation & Correctness
* **Metric:** Because survival data includes censored patients (patients who did not experience the event during the study), standard "Accuracy" cannot be used.
* **Solution:** The pipeline's correctness will be mathematically evaluated using the **Concordance Index (C-Index)**, which measures the rank correlation between the predicted Cox-PH risk scores and the actual survival times across patient pairs.
