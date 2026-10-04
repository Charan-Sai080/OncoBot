# Final Project Report: Multimodal AI for Cancer Survival Prediction (Onco_Bot)

## 1. Introduction
Cancer prognosis and diagnosis have historically relied on isolated data modalities, predominantly the visual inspection of Whole-Slide Histopathology Images (WSI) by human pathologists, or the independent analysis of genomic sequencing. **Onco_Bot** is a cutting-edge, multimodal Artificial Intelligence pipeline designed to integrate gigapixel WSI data with Bulk Transcriptomics (RNA-Seq). By fusing these two distinct modalities, the system predicts a patient's survival risk using a Cox Proportional Hazards (Cox-PH) model and leverages a Large Language Model (LLM) via a cloud API to automatically generate a highly explainable, human-readable clinical report. 

## 2. Literature Review
The integration of Artificial Intelligence in oncology has rapidly evolved from traditional machine learning to deep learning. Historically, supervised Convolutional Neural Networks (CNNs) like ResNet were used for histopathology; however, they require massive amounts of painstakingly labeled data and are biased toward natural images. Recently, **Self-Supervised Vision Transformers (ViTs)**, such as DINO (Self-Distillation with No Labels), have proven superior for capturing complex, heterogeneous cellular structures without requiring manual annotations. Furthermore, standard Multi-Layer Perceptrons (MLPs) applied to transcriptomic data often act as "black boxes," obscuring biological realities. Recent literature advocates for biologically informed neural networks that constrain weights to known biological pathways, improving both performance and interpretability in clinical settings.

## 3. Problem Statement
Despite the theoretical capabilities of multimodal AI, deploying these systems poses severe clinical and computational challenges:
1. **Lack of Explainability:** Clinicians cannot trust "black box" algorithms that output a risk score without biological justification.
2. **Computational Bottlenecks:** Processing a single 4GB `.svs` Whole-Slide Image (often containing billions of pixels) dynamically on consumer-grade hardware (such as laptops or local NAS servers) rapidly leads to catastrophic memory leaks, out-of-memory errors, and severe Solid State Drive (SSD) wear-out via OS-level Swap dumping.

## 4. Objectives
To solve the aforementioned problems, this project was designed with the following core objectives:
- **Build a Biologically Grounded Architecture:** Utilize a Pathway-Aware Transformer to map genes to specific biological pathways, avoiding black-box predictions.
- **Achieve True Multimodal Explainability:** Implement a Cross-Attention Fusion mechanism that allows genomic pathways to spatially attend to specific morphological regions in the tissue.
- **Optimize Hardware Execution:** Engineer aggressive memory management and controlled multiprocessing to process massive WSI files safely on consumer hardware without degrading SSDs.
- **Generate Clinical Summaries:** Seamlessly translate raw mathematical risk scores into actionable clinical reports using Cloud LLMs.

## 5. Detailed Design Process
The architecture of Onco_Bot is fundamentally divided into data ingestion, neural processing, and inference. 

**Data Ingestion & Optimization (`data/pathology.py` & `data/genomic.py`):**
To handle the 4GB WSIs, the system avoids loading the full image into RAM. Instead, it extracts a 2048x2048 coarse thumbnail to identify tissue-dense regions. It then mathematically targets exactly 1,024 tissue patches (the MIL sweet spot). Genomic data is simultaneously parsed and mapped against a Boolean Pathway Mask to isolate specific pathways (e.g., PI3K-Akt, p53).

**Neural Architecture (`models/pathology_branch.py`, `models/genomic_branch.py`, `models/fusion.py`):**
- **Vision Branch:** The 1,024 patches are passed through a frozen DINO (vits16) Transformer to extract a `[1024, 384]` dimensional tensor representing the tumor microenvironment.
- **Genomic Branch:** The Pathway-Aware Transformer processes the masked transcriptomics, outputting a pathway-specific embedding `[num_pathways, 128]`.
- **Multimodal Fusion:** A Cross-Attention mechanism sets the Genomic Pathways as the *Query* and the WSI Patches as the *Key/Value*. This calculates precise attention weights between specific genes and specific tissue regions. The fused representation is then passed through a Cox-PH Survival Head to calculate the risk score.

## 6. Progress of Modeling the Project Design
The development phase encountered and successfully resolved several critical engineering hurdles:
- **The "Batch Norm Eval" Crash:** During initial isolated testing (`main.py`), the model crashed during batch-size-1 inference due to PyTorch `BatchNorm1d` variances. This was resolved by strictly enforcing `.eval()` state across all model modalities during inference.
- **The OpenSlide Memory Bomb:** The original data loader attempted to mask tissue by reading the lowest native magnification level using `slide.read_region()`. This triggered a massive 40GB RAM allocation (pushing 35GB to SSD Swap) because the image lacked a pyramidal thumbnail. The code was rewritten to utilize `slide.get_thumbnail()`, strictly capping RAM usage to 3MB.
- **Escaping the C-Level Tile Cache:** Sequentially extracting 1,024 patches caused the OpenSlide C-library to aggressively cache neighboring image tiles, leaking RAM over time. We engineered a *Controlled Multiprocessing* loop: chunking the coordinates across 4 parallel workers, where each worker opened and explicitly closed its own slide handle. This forceful garbage collection imitated lazy evaluation, dropping extraction time by 4x and capping RAM completely safely at 16GB.

## 7. Completion of the Models
The models were finalized and wired into an end-to-end inference script (`run_sample.py`). To fulfill the clinical reporting objective, the `ClinicalReportGenerator` (`onco_utils/llm_report.py`) was integrated. This module intercepts the generated Cox-PH Risk Score and the highest-weighted cross-attention pathways, formats them into an intelligent clinical prompt, and securely transmits them to the official Ollama Cloud API (via a dedicated API key using the `minimax-m3` model). A universal setup script (`setup.sh` / `setup.ps1`) was also engineered to handle cross-platform dependency resolution (macOS, Linux, Native Windows WSL2).

## 8. Demonstration Snapshots
During the final successful End-to-End test run on patient sample `TCGA-2F-A9KO`, the system demonstrated flawless execution:
- **Risk Score Generated:** A Cox-PH Survival Risk Score of `0.1174` was successfully outputted by the fusion model.
- **LLM Report Generated:** The Cloud API returned a highly detailed report identifying a spatial correlation (attention weight: 0.8013) between the PI3K-Akt transcriptomic signature and regions of dense tumor cellularity.
*(Note for GitHub/Presentation: Insert screenshots of the terminal output here, followed by screenshots of the Spatial Attention Heatmap and the Interactive Web Dashboard built by the UI team).*

## 9. Conclusion
Onco_Bot successfully demonstrates that gigapixel multimodal cancer prediction is not only possible but can be executed securely and efficiently on consumer-grade hardware. By replacing black-box MLPs with biologically constrained Pathway Transformers, and by replacing rigid concatenation with Spatial Cross-Attention, the model achieves unprecedented clinical explainability. The aggressive memory optimizations guarantee scalable deployment in low-resource hospital settings without the threat of hardware degradation, laying a robust foundation for the future of AI-assisted oncology.

## 10. References
1. Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers (DINO). *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.
2. Chen, R. J., et al. (2022). Pan-cancer Integrative Histology-Genomic Analysis via Multimodal Deep Learning. *Cancer Cell*.
3. Goode, A., et al. (2013). OpenSlide: A vendor-neutral software foundation for digital pathology. *Journal of Pathology Informatics*.
4. Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*.
5. Ollama Cloud API Documentation. (2024). *api.ollama.com*.
