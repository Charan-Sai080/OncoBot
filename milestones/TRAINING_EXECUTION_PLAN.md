# Milestone: Execute Distributed Training

This document outlines the exact 3-phase execution plan to complete the training of the Onco_Bot multimodal network by the end of the day (EOD).

## Phase 1: Feature Extraction (Cloud / Google Colab)
**Objective:** Convert heavy 4GB Whole-Slide Images (WSIs) into lightweight mathematical tensors without crashing local hardware.
- [ ] Mount Google Drive in Google Colab Pro.
- [ ] Upload the 403 `.svs` Bladder Cancer WSIs to the cloud.
- [ ] Run a Python loop to extract 1,024 patches per image using OpenSlide.
- [ ] Pass the patches through the pre-trained **DINO Vision Transformer** (vits16).
- [ ] Save the 403 outputs as `.pt` feature embedding files (approx. 1.5MB each).

## Phase 2: Data Transfer (Cloud to Edge)
**Objective:** Move the mathematical representations back to the local Apple Silicon Mac for fusion training.
- [ ] Download the 403 `.pt` files from Google Drive.
- [ ] Place all `.pt` files directly into the local `Datasets/WSI_Features/` folder.
- [ ] Ensure `Datasets/TCGA-BLCA.star_counts.tsv` (Genomics), `Datasets/aligned_3way_slides_MINIMAL_case_level.csv` (Patient Labels), and `Datasets/kegg_cancer_pathways.gmt` (KEGG Mask) are present.

## Phase 3: Local Fusion Training (Mac)
**Objective:** Train the Cross-Attention Fusion module and Cox-PH Survival Head to learn Bladder Cancer patterns.
- [ ] Open the terminal and activate the conda environment: `conda activate onco_bot`
- [ ] Execute the training script: 
  ```bash
  KMP_DUPLICATE_LIB_OK=TRUE python main.py
  ```
- [ ] Monitor the training loop to ensure the Loss decreases over epochs.
- [ ] Upon completion, the model weights will be optimized and the pipeline will be ready for mathematically and biologically accurate inference.
