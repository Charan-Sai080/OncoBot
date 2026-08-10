# Onco_Bot: Technical Journey & Roadmap

This document outlines the detailed, step-by-step engineering journey we will undertake to build, train, and deploy the Onco_Bot (Pathway-Aware Multimodal Transformer) for Bladder Cancer.

## Phase 1: The Genomic Branch (Foundation)
Our journey begins with the patient's genetic data. Raw gene expression data is noisy and difficult for neural networks to interpret meaningfully.

1.  **Data Acquisition**: We download the TCGA-BLCA RNA-seq dataset and clinical survival labels from UCSC Xena.
2.  **KEGG Pathway Mapping**: Instead of feeding ~60,000 raw genes into a model, we map them to 186 known biological pathways (using the KEGG database). This groups genes by their actual biological functions (e.g., cell apoptosis, DNA repair).
3.  **Pathway Tokenization**: We aggregate the expression values of genes within each pathway to create dense "pathway tokens".
4.  **Self-Attention Encoding**: These tokens are passed through a Transformer Encoder. The self-attention mechanism learns how different biological pathways interact and influence each other in bladder cancer patients.

## Phase 2: The Pathology Branch (Vision)
Next, we process the physical evidence of the tumor using Whole-Slide Images (WSIs).

1.  **Efficient Data Loading**: To overcome hardware limitations, we will build a pipeline that processes one `.svs` WSI at a time, avoiding storing hundreds of gigabytes of raw images.
2.  **Tissue Segmentation & Patch Extraction**: Using `OpenSlide`, we segment the actual tissue from the blank glass background and extract small (e.g., 256x256) image patches.
3.  **Patch Clustering**: A WSI can have 10,000+ patches. We use K-Means clustering to group them by visual similarity and sample 10 patches from 50 clusters, resulting in exactly 500 highly representative patches per patient.
4.  **DINO Feature Extraction**: We pass these 500 patches through a pre-trained Vision Transformer (DINO). We save the output embeddings (which are tiny in file size) and delete the raw WSI.
5.  **Self-Attention Encoding**: Similar to the genes, the 500 patch embeddings are passed through a Vision Transformer Encoder to learn spatial and morphological interactions across the tumor.

## Phase 3: Multimodal Fusion & Alignment (The Core Innovation)
This is where we combine biology (genes) with morphology (images).

1.  **Label-Free Contrastive Alignment**: Before fusing, the model must understand that genes and images live in the same "semantic space." We use an InfoNCE contrastive loss to pull the most correlated pathway tokens and patch tokens closer together.
2.  **Cross-Attention Fusion**: We use a Cross-Attention Transformer. The *Genomic Pathways* act as the **Query**, while the *Image Patches* act as the **Key and Value**. This mathematically forces the model to look at the image *through the lens* of the genetic pathways, answering the question: "Where is this genetic mutation manifesting in the physical tissue?"
3.  **Survival Prediction Head**: The fused multimodal representation is passed through a Multi-Layer Perceptron (MLP). We optimize the network using a **Cox Proportional Hazards Loss** (Negative Partial Log-Likelihood) to predict the patient's survival risk score.

## Phase 4: Interpretability & Novelty (The Product)
Finally, we transform the mathematical model into a clinical tool.

1.  **Extracting Attention Maps**: Because we used cross-attention, we can extract the exact attention weights. This tells us exactly which genetic pathway was paying attention to which specific physical patch of the tumor.
2.  **LLM Clinical Report Generation**: We extract the top 3 highest-risk genetic pathways and the top 5 most dangerous tumor patches for a patient. We feed this structured data to a Large Language Model (e.g., via API) with a custom prompt to generate a human-readable clinical pathology report.
3.  **Interactive Dashboard**: We build a web application (using Streamlit). The user uploads a patient's data, and the dashboard displays the predicted survival curve, overlays the attention heatmaps onto the WSI to highlight dangerous regions, and displays the LLM-generated clinical report.

## Evaluation & Metrics
Throughout the journey, we will evaluate the model using:
*   **Concordance Index (C-index)**: To measure if the model correctly ranks patients by survival time.
*   **Kaplan-Meier Curves**: To prove the model can successfully split patients into distinct "High Risk" and "Low Risk" cohorts.
