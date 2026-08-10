# Onco_Bot: Multimodal AI for Cancer Survival Prediction
## Technical Case Study & Interview Prep Guide

This document provides a deep dive into the architecture, engineering decisions, and problem-solving strategies used to build **Onco_Bot**. It is designed to serve as a technical portfolio piece for GitHub and a study guide to defend the project in front of academic panels or technical interviewers.

---

## 1. System Architecture & Data Flow

Onco_Bot is a multimodal deep learning pipeline that fuses two highly disparate data types—Gigapixel Whole-Slide Images (WSI) and Bulk Transcriptomics (RNA-Seq)—to predict patient survival risk and generate explainable clinical reports.

### The Pipeline Data Flow:
1. **Input Phase:**
   - **Pathology (WSI):** A massive `.svs` file (often 4GB+) containing billions of pixels.
   - **Genomics:** A `.csv` file containing normalized gene expression values.
2. **Feature Extraction Phase:**
   - **Vision Branch:** The WSI is randomly sampled for 1,024 patches (the "sweet spot" for Multiple Instance Learning). These patches are passed through a frozen **DINO Vision Transformer**, resulting in a `[1024, 384]` feature tensor.
   - **Genomic Branch:** Gene expressions are mapped to known biological pathways using a boolean mask. A **Pathway-Aware Transformer** processes this, yielding a `[num_pathways, 128]` tensor.
3. **Multimodal Fusion Phase:**
   - A **Cross-Attention Mechanism** is employed. The *Genomic Pathways* act as the Query (Q), and the *WSI Patches* act as the Key/Value (K, V).
   - *Why?* This allows the model to ask: "Given this upregulated pathway, which specific regions of the tissue morphology are most relevant?"
4. **Prediction & Explainability Phase:**
   - The fused tensor is passed through an MLP to output a **Cox-PH Risk Score**.
   - The Cross-Attention weights are extracted to identify the top activated pathways and their corresponding tissue patches.
5. **LLM Generation Phase:**
   - The Risk Score and top pathway/morphology relationships are structured into a prompt and sent to an **Ollama Cloud API** (running Qwen/Minimax), which returns a human-readable clinical executive summary.

---

## 2. Key Engineering & Coding Decisions

### Why DINO for the Vision Encoder?
Traditional supervised CNNs (like ResNet trained on ImageNet) are biased towards natural images (dogs, cars). **DINO (Self-Distillation with No Labels)** is a self-supervised Vision Transformer. It learns to understand local and global feature structures without needing labeled data, making it vastly superior for capturing complex, heterogeneous cellular structures in histopathology.

### Why a "Pathway-Aware" Transformer instead of a standard MLP?
Feeding 20,000 raw genes into a standard Multi-Layer Perceptron creates a "black box" that is prone to overfitting and biologically uninterpretable. By mapping genes to strict biological pathways (e.g., PI3K-Akt, p53) before passing them through the Transformer, we inject *biological priors* into the architecture. This forces the model to learn at the pathway level, making the final predictions highly explainable to oncologists.

### Why Cross-Attention Fusion instead of simple Concatenation?
Simple concatenation (flattening the genomic and vision tensors and joining them) destroys the spatial structure of the WSI and the discrete nature of the pathways. Cross-Attention preserves the dimensional integrity of both modalities, mathematically forcing the model to explicitly calculate the correlation between specific genes and specific tissue regions.

---

## 3. Major Challenges Faced & Solutions Implemented

### Challenge 1: The "Batch Norm Eval" Crash
**The Problem:** During the initial pipeline test on a single sample patient, the model crashed with a PyTorch error regarding `BatchNorm1d` expecting more than 1 value.
**The Root Cause:** In PyTorch, Batch Normalization layers calculate running means and variances across a batch of data. During inference, if the batch size is exactly `1` (one patient) and the model is still in `train()` mode, the BatchNorm layer fails because it cannot calculate a variance for a single number.
**The Solution:** We explicitly appended `.eval()` to all model initializations (`genomic_model.eval()`, `pathology_model.eval()`) in the inference script. This locks the BatchNorm layers into using their pre-calculated moving averages, successfully enabling batch-size-1 inference.

### Challenge 2: The OpenSlide "Memory Bomb" (35GB Swap Leak)
**The Problem:** While trying to randomly sample patches, the system's memory violently spiked, pushing 35GB of data into the Mac's SSD Swap, heavily lagging the system and degrading the NAND flash.
**The Root Cause:** To figure out where tissue existed, the code read the lowest magnification level of the WSI to create a tissue mask using `slide.read_region()`. Because the specific `.svs` file lacked a pre-computed thumbnail pyramid, `read_region` attempted to load a 50,000 x 50,000 pixel array directly into RAM—a 40 Gigabyte allocation.
**The Solution:** We patched the data loader to use `slide.get_thumbnail((2048, 2048))`. This forces the underlying OpenSlide C-library to dynamically downsample the image directly on the disk, returning a safe, capped array that consumes only 3 Megabytes of RAM.

### Challenge 3: Escaping the C-Level Tile Cache (The "Lazy" Imitation)
**The Problem:** Even after fixing the memory bomb, sequentially reading 1,024 patches across a 4GB image was slow, and OpenSlide's aggressive C-level tile caching mechanism kept hoarding gigabytes of memory to anticipate future reads.
**The Root Cause:** `openslide-python` does not expose an API to disable the internal C-cache. Passing a single `OpenSlide` object across a loop allows the cache to accumulate indefinitely.
**The Solution:** We engineered a **Controlled Multiprocessing** architecture. We chunked the 1,024 coordinates into 4 tiny batches and spawned 4 parallel CPU workers. Crucially, each worker opened its *own* slide handle, processed its 256 patches, and explicitly executed `slide.close()`. 
**The Result:** By closing the slide handle, the C-level memory was violently destroyed before it could accumulate. This imitated "lazy evaluation", dropping extraction time from 2 minutes to 15 seconds while perfectly capping system RAM usage at a safe 16GB with zero Swap.

---

## 4. Potential Panel Questions & How to Answer Them

**Q: "Why did you choose 1,024 patches? Doesn't that miss a lot of the tumor?"**
*Answer:* "In Multiple Instance Learning (MIL) for pathology, processing the entire 100,000x100,000 image is computationally impossible. 1,024 patches is an industry-standard 'sweet spot'. It provides enough morphological variance to capture the tumor microenvironment (stroma, necrosis, lymphocytes) while fitting inside modern GPU memory limits during training."

**Q: "How does your model handle the massive size of WSI files on consumer hardware like a Mac?"**
*Answer:* "We implemented a custom data loader that avoids loading the full image into RAM. We use `get_thumbnail` for masking, and we utilize a Controlled Multiprocessing approach where worker threads open independent OpenSlide handles, extract small chunks of patches, and close the handles immediately. This forcefully garbage-collects the C-level tile cache, keeping our RAM footprint below 16GB and eliminating SSD swap wear-out."

**Q: "Is your AI a 'Black Box'?"**
*Answer:* "No. Interpretability is baked into the architecture. By using a Pathway-Aware Transformer, we map genes to known biological pathways. By using Cross-Attention, we can literally visualize which tissue patches the model was looking at when it prioritized a specific pathway. The upcoming Spatial Attention Heatmap feature will visually project these attention weights back onto the original WSI."
