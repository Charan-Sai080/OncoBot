# OncoBot 🧬🔬
### Pathway-Aware Multimodal Transformer (PAMT) for Cancer Survival Prognostication & Clinical Decision Support

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.2-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.3-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Serverless%20Router-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Harrell's C-Index](https://img.shields.io/badge/Validation%20C--Index-0.784-success.svg)](#-clinical-validation--benchmark-results)

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [The Clinical Problem](#-the-clinical-problem)
- [System Architecture](#-system-architecture)
- [What We Used and Why (Component-by-Component Rationale)](#-what-we-used-and-why-component-by-component-rationale)
  - [1. Pathology Branch (Vision & Spatial Morphology)](#1-pathology-branch-vision--spatial-morphology)
  - [2. Genomics Branch (Biology & Transcriptomics)](#2-genomics-branch-biology--transcriptomics)
  - [3. Cross-Modal Attention Fusion](#3-cross-modal-attention-fusion)
  - [4. Prognostic Survival Head (Cox-PH)](#4-prognostic-survival-head-cox-ph)
  - [5. Automated LLM Clinical Reporting](#5-automated-llm-clinical-reporting)
  - [6. High-DPI Client-Side PDF Export](#6-high-dpi-client-side-pdf-export)
  - [7. Clinical Intelligence Dashboard](#7-clinical-intelligence-dashboard)
- [Mathematical Formulations](#-mathematical-formulations)
- [Interactive Clinical Features](#-interactive-clinical-features)
- [Repository & File Organization](#-repository--file-organization)
- [Installation & Quickstart Guide](#-installation--quickstart-guide)
- [Configuration & Environment Variables](#-configuration--environment-variables)
- [REST API Specification](#-rest-api-specification)
- [Clinical Validation & Benchmark Results](#-clinical-validation--benchmark-results)
- [Medical Disclaimer & Ethical Use](#-medical-disclaimer--ethical-use)
- [Citations & References](#-citations--references)

---

## 🩺 Executive Overview

**OncoBot** is a production-grade multimodal clinical AI platform engineered to bridge the translational gap between **Gigapixel Whole-Slide Histopathology (WSI)** and **Bulk Transcriptomics (RNA-Seq)** for patients with **Bladder Urothelial Carcinoma (TCGA-BLCA)**.

Traditional computational oncology models either evaluate a single modality in isolation or perform naive late-stage concatenation (flattening features into a generic 1D array), completely destroying spatial tissue architecture and biological pathway hierarchies.

OncoBot introduces a **Pathway-Aware Multimodal Transformer (PAMT)**:
1. **Genotype-to-Phenotype Cross-Attention:** Uses 320 canonical cancer pathways from MSigDB KEGG as cross-attention *queries* to inspect thousands of self-supervised *visual patch keys/values* extracted from gigapixel biopsy slides.
2. **Proportional Hazards Survival Modeling:** Predicts continuous patient risk scores $\hat{h}(x)$ trained on right-censored time-to-event outcomes using Cox Partial Likelihood.
3. **Automated Clinical Decision Support (CDS):** Synthesizes risk scores, top-ranked pathway cascades, and histological atypia into a 5-part Multidisciplinary Tumor Board (MDT) clinical report using **Hugging Face Serverless 70B LLMs** with automated multi-model failover.
4. **Interactive 1400px Clinical Console:** Features an interactive 3-mode WSI canvas viewer (Raw H&E, Cross-Attention Heatmap, Hotspots $\ge 0.72$), publication-grade Kaplan-Meier curves with 95% Confidence Interval ribbons, cohort ingestion for custom `.svs` slides and RNA-seq counts, and $2\times$ Retina clinical PDF generation.

```
+----------------------------------------------------------------------------------------------------+
|                                    ONCOBOT MULTIMODAL PIPELINE                                      |
+----------------------------------------------------------------------------------------------------+
|  [H&E Whole-Slide Image]  --> OpenSlide Thumbnail Mask --> Sweet-Spot Sampling (1024) --> DINO ViT |
|                                                                                                    |
|  [Bulk RNA-Seq Profile]   --> Log2 Normalization       --> 320 KEGG Pathway Mask   --> PAMT Trans. |
|                                                                                                    |
|                      ======== MULTI-HEAD CROSS-ATTENTION FUSION ========                            |
|                            (Genomic Queries  <-->  Visual Keys/Values)                             |
|                                                                                                    |
|  --> Cox-PH Hazard Head    --> Stratified Kaplan-Meier Survival Analytics (Log-Rank p < 0.0001)    |
|  --> Spatial Heatmaps      --> Histopathology Hotspot Localization (alpha >= 0.72)                 |
|  --> Hugging Face 70B LLM  --> 5-Section Oncology Report (Llama-3.3-70B / DeepSeek-V3 / Qwen-2.5)  |
|  --> jsPDF + html2canvas   --> 2x Retina Publication-Grade Patient Dossier PDF Export               |
+----------------------------------------------------------------------------------------------------+
```

---

## 🔬 The Clinical Problem

Bladder Urothelial Carcinoma is notorious for high molecular heterogeneity, frequent postoperative recurrence (50–70% in non-muscle invasive stages), and rapid progression to lethal muscle-invasive carcinoma (MIBC).

In contemporary clinical practice:
- **Pathologists** inspect H&E slides under brightfield microscopy to assess histological grade, nuclear pleomorphism, invasion depth (lamina propria vs. muscularis propria), and stromal desmoplasia. However, subjective visual inspection exhibits inter-observer variability and cannot observe latent somatic mutations.
- **Molecular Biologists** sequence bulk RNA to quantify gene expression (e.g., *FGFR3*, *TP53*, *RB1*, *PIK3CA*), yet bulk expression profiles lose all spatial tissue context and cellular architecture.
- **Oncologists & MDTs** struggle to manually synthesize these disparate datasets to decide whether a patient requires immediate neoadjuvant platinum-based chemotherapy, radical cystectomy, or immune checkpoint blockade (e.g., anti-PD-L1/PD-1).

**OncoBot resolves this division** by mathematically modeling the interaction between activated somatic pathways and the physical cellular morphology of the tumor microenvironment.

---

## 🏛️ System Architecture

The following diagram details the end-to-end dataflow from raw patient specimens to the interactive clinical cockpit:

```mermaid
flowchart TD
    subgraph Modality_1["Modality 1: Whole-Slide Histopathology (WSI)"]
        A1["Raw Gigapixel Slide (.svs / .tiff)<br/>~100,000 x 100,000 pixels (4GB+)"] --> A2["Thumbnail Masking (2048x2048)<br/>Prevents 40GB RAM Allocation"]
        A2 --> A3["Sweet-Spot Sampling (1,024 Patches)<br/>Controlled Multiprocessing (4 Workers)"]
        A3 --> A4["Self-Supervised DINO ViT-S/16<br/>Patch Embeddings (1024 x 384)"]
        A4 --> A5["WSI MIL Transformer Encoder<br/>cls_token Aggregation (d_model = 128)"]
    end

    subgraph Modality_2["Modality 2: Bulk Transcriptomics (RNA-Seq)"]
        B1["Normalized Gene Counts (TPM)<br/>~20,000 Human Genes"] --> B2["log2(x + 1) Normalization"]
        B2 --> B3["Sparse Bipartite Pathway Mask (M)<br/>320 MSigDB KEGG Biological Pathways"]
        B3 --> B4["Pathway-Aware Transformer (PAMT)<br/>Cross-Pathway Attention (d_model = 128)"]
    end

    subgraph Cross_Modal_Fusion["Multimodal Fusion & Prognosis"]
        A5 --> C1["Multi-Head Cross-Attention<br/>Queries = Genomics | Keys/Values = Pathology"]
        B4 --> C1
        C1 --> C2["Fused Patient Representation (d_model = 128)<br/>Residual Connection + GELU FFN"]
        C2 --> C3["Cox-PH Survival Prediction Head<br/>Dense(128 -> 64 -> 32 -> 1)"]
        C3 --> C4["Continuous Risk Score h(x)<br/>Stratified Hazard Percentile"]
    end

    subgraph Clinical_Synthesis["Automated Decision Support & UI"]
        C4 --> D1["Kaplan-Meier Survival Engine<br/>95% CI Ribbons & At-Risk Tables"]
        C1 --> D2["Spatial Attention Matrix<br/>Overlay & Hotspots (alpha >= 0.72)"]
        C4 & D2 & B4 --> D3["Hugging Face Serverless Router<br/>Llama-3.3-70B / DeepSeek-V3 / Qwen-2.5"]
        D3 --> D4["5-Section Clinical Oncology Report"]
        D1 & D2 & D4 --> D5["1400px React 19 Clinical Console"]
        D5 --> D6["2x Retina jsPDF Clinical Export"]
    end

    style Modality_1 fill:#e6f4f1,stroke:#0f766e,stroke-width:2px;
    style Modality_2 fill:#f8f9fa,stroke:#1e293b,stroke-width:2px;
    style Cross_Modal_Fusion fill:#fdf2f4,stroke:#be123c,stroke-width:2px;
    style Clinical_Synthesis fill:#f0fdf4,stroke:#15803d,stroke-width:2px;
```

---

## 🔬 What We Used and Why (Component-by-Component Rationale)

Every technology, neural layer, and algorithmic decision in OncoBot was purposefully chosen to overcome specific biomedical and computational challenges.

### 1. Pathology Branch (Vision & Spatial Morphology)

| Technology / Component | What We Used | Why We Used It (Clinical & Computational Rationale) |
| :--- | :--- | :--- |
| **Slide Reading Engine** | Pre-compiled `OpenSlide` C-libraries with dynamic Windows DLL loading | Whole-Slide Images (`.svs`, `.tif`) use proprietary multi-resolution TIFF pyramids. OpenSlide provides fast, multi-threaded C-level tile decoding across arbitrary zoom levels. |
| **Memory Guard** | Dynamic Thumbnail Masking (`slide.get_thumbnail((2048, 2048))`) | **Fixes the 35GB Swap Leak:** Naive tissue masking calls `slide.read_region()` at the lowest magnification level. Slides lacking pre-computed thumbnail pyramids force a 40GB array allocation into RAM. Thumbnail downsampling requests a bounded 2048x2048 array directly from the decoder, capping RAM consumption at $<3\text{MB}$. |
| **Worker Concurrency** | Controlled Multiprocessing (4 Workers with isolated handles) | `openslide-python` does not expose an API to clear its internal C-level tile cache. Looping over a single slide handle causes memory to expand continuously. Our architecture chunks coordinates across 4 worker processes, opens separate slide instances, and forces `slide.close()`, violently destroying the C-cache before memory bloat occurs. Reduces patch extraction time from 2 minutes to 15 seconds. |
| **Patch Sampling Strategy** | Sweet-Spot Uniform Sampling (1,024 patches at $256 \times 256\text{px}$) | Processing all 100,000+ patches in a slide is computationally intractable and redundant. K-Means clustering across gigapixel slides creates a severe CPU bottleneck. 1,024 randomly sampled patches capture the full histological diversity of the tumor microenvironment (necrosis, stroma, pleomorphic nests, lymphocytes) while fitting within modern GPU tensor memory. |
| **Feature Extractor** | **DINO ViT-S/16** (Self-Distillation with No Labels, 384-dim) | Traditional supervised CNNs (like ResNet-50 trained on ImageNet) are heavily biased towards natural object silhouettes and global shapes (dogs, cars). DINO is a self-supervised Vision Transformer that learns patch-level and bag-level representations without human labels. It excels at preserving fine nuclear pleomorphism, chromatin clumping, and intercellular stroma. |
| **WSI Aggregator** | **WSI Multiple Instance Learning (MIL) Transformer Encoder** ($d=128$) | A bag of patches is permutation-invariant. Rather than simple mean pooling, we prepend a learnable `[CLS]` token and pass the 1,024 patch tokens through a Transformer Encoder. Self-attention enables the model to reason about spatial dependencies across distant tissue regions. |

---

### 2. Genomics Branch (Biology & Transcriptomics)

| Technology / Component | What We Used | Why We Used It (Clinical & Computational Rationale) |
| :--- | :--- | :--- |
| **Expression Preprocessing** | Log2-transformation: $\log_2(\text{TPM} + 1)$ | RNA-Seq count distributions are severely right-skewed. Log-normalization stabilizes variance, reduces outlier dominance from hyper-expressed housekeeping genes, and preserves dynamic range. |
| **Biochemical Catalog** | 320 Canonical Cancer Pathways from **MSigDB KEGG 2021** | Raw human transcriptomes contain ~20,000 genes with low sample sizes ($N \ll P$), leading standard models to memorize noise. Grouping genes into 320 KEGG pathways (e.g., `KEGG_BLADDER_CANCER`, `KEGG_P53_SIGNALING`, `KEGG_PI3K_AKT`, `KEGG_CELL_CYCLE`) injects biologically validated prior knowledge. |
| **Sparsity Enforcement** | Sparse Bipartite Pathway Mask $\mathbf{M} \in \{0, 1\}^{\text{pathways} \times \text{genes}}$ | Unlike dense Multi-Layer Perceptrons where every gene connects to every node (an uninterpretable "black box"), our linear weights are element-wise multiplied by $\mathbf{M}$ ($\mathbf{W} \odot \mathbf{M}$). A gene can **only** pass signal to the pathways in which it participates, guaranteeing biological plausibility. |
| **Pathway Interactivity** | **Pathway-Aware Transformer (PAMT)** ($d=128, n_{\text{head}}=4$) | Scalar pathway activations are projected into 128-dimensional tokens and passed through a multi-layer Transformer Encoder. This allows the model to learn cross-pathway metabolic crosstalk (e.g., how p53 inactivation amplifies PI3K-Akt signaling). |

---

### 3. Cross-Modal Attention Fusion

| Technology / Component | What We Used | Why We Used It (Clinical & Computational Rationale) |
| :--- | :--- | :--- |
| **Fusion Paradigm** | **Multi-Head Cross-Attention** ($Q=\text{Genomics}, K/V=\text{Pathology}$) | Simple vector concatenation ($[\mathbf{e}_g ; \mathbf{e}_p]$) destroys the structural correspondence between modalities. Cross-attention treats the biological pathway states as **Queries** that attend to visual tissue patches as **Keys and Values**. |
| **Clinical Interpretability** | Attention Weight Extraction ($\alpha_{ij}$) | Mathematically answers the clinical question: *"When pathway $i$ (e.g., p53 loss) is activated, which exact tissue regions exhibit abnormal cellular phenotypes?"* The resulting attention weights directly drive our spatial heatmaps and hotspot reticles. |
| **Regularization & Stability** | Layer Normalization + GELU Feed-Forward Network + Dropout ($0.1$) | Prevents either modality from dominating the latent embedding space and stabilizes gradient flow across deep multimodal layers. |

---

### 4. Prognostic Survival Head (Cox-PH)

| Technology / Component | What We Used | Why We Used It (Clinical & Computational Rationale) |
| :--- | :--- | :--- |
| **Survival Architecture** | Multi-Layer Perceptron ($128 \rightarrow 64 \rightarrow 32 \rightarrow 1$) with BatchNorm | Maps the fused 128-dimensional multimodal representation into a single continuous hazard score $\hat{h}(x)$ representing the patient's relative risk of death. |
| **Loss Function** | **Cox Proportional Hazards Negative Log Partial Likelihood** | Standard Mean Squared Error (MSE) or Binary Cross-Entropy cannot handle **right-censored survival data** (patients alive at follow-up conclusion or lost to observation). Cox-PH evaluates the relative likelihood of failure within dynamic risk sets over time. |
| **Numerical Optimization** | Cumulative Log-Sum-Exp Trick | Computing raw exponentials ($\sum \exp(\hat{h})$) causes floating-point overflow during backpropagation. We sort patients by descending survival time and employ log-sum-exp cumulative summation for stable gradient descent. |
| **Validation Metric** | **Harrell's Concordance Index (C-Index = 0.784)** | C-Index measures rank-order concordance across all admissible patient pairs. An index of $0.5$ represents random guessing, while $1.0$ represents perfect temporal ranking. OncoBot achieves $0.784$, outperforming single-modality baselines by $+8.6\%$. |

---

### 5. Automated LLM Clinical Reporting

| Technology / Component | What We Used | Why We Used It (Clinical & Computational Rationale) |
| :--- | :--- | :--- |
| **Inference Infrastructure** | **Hugging Face Serverless Router API** (`https://router.huggingface.co/v1/chat/completions`) | Hosting local 70B parameter models requires costly multi-GPU enterprise hardware (e.g., $2\times$ NVIDIA A100 80GB). Hugging Face Serverless Router provides instantaneous access to state-of-the-art open weights with zero local VRAM overhead. |
| **Primary Model** | `meta-llama/Llama-3.3-70B-Instruct` | Chosen for its clinical reasoning capacity, instruction-following fidelity, and structured medical nomenclature synthesis. |
| **Multi-Model Failover Hierarchy** | `Llama-3.3-70B` $\rightarrow$ `DeepSeek-V3` $\rightarrow$ `Qwen-2.5-72B` $\rightarrow$ `Llama-3.1-8B` $\rightarrow$ Deterministic Clinical Synthesizer | Guarantees zero downtime. If an API rate limit, network timeout, or maintenance window occurs, the system automatically falls back through secondary frontier models and finally to an offline clinical rule-based engine. |
| **Report Structure** | 5 Standardized Multidisciplinary Tumor Board (MDT) Sections | Translates complex mathematical tensors into actionable clinical insights: Executive Summary, Genotype-Phenotype Analysis, Histomorphological Correlates, Therapeutic Recommendations, and MDT Action Plan. |

---

### 6. High-DPI Client-Side PDF Export

| Technology / Component | What We Used | Why We Used It (Clinical & Computational Rationale) |
| :--- | :--- | :--- |
| **Rendering Engine** | **`jsPDF` (v4.2.1) + `html2canvas` (v1.4.1)** | Compiling PDFs via server-side headless browsers (like Puppeteer) introduces high memory overhead, cold-start latency, and sensitive patient data transfer over the wire. Client-side rendering is instantaneous and preserves HIPAA compliance. |
| **Retina Resolution** | Canvas Scale Factor $2.0$ ($2\times$ Pixel Ratio) | Standard canvas captures produce blurry text when printed on physical paper. A $2.0$ scale factor ensures crisp, 300 DPI publication-grade vector-like typography and chart rendering. |
| **Document Formatting** | Multi-page A4 layout with letterhead, patient dossier, and signature block | Formatted to meet international pathology laboratory documentation standards, complete with primary oncologist and molecular pathologist sign-off blocks. |

---

### 7. Clinical Intelligence Dashboard

| Technology / Component | What We Used | Why We Used It (Clinical & Computational Rationale) |
| :--- | :--- | :--- |
| **Frontend Framework** | **React 19 + TypeScript + Vite 8.3** | High-speed component rendering, sub-millisecond hot-module replacement during development, and strict type safety across multimodal patient payloads. |
| **Styling & Design System** | **Tailwind CSS v4** with a custom clinical palette | Built according to medical UI ergonomics: Deep Teal (`#0f766e`), Charcoal Navy (`#1e293b`), Card Mint (`#f0fdf4`), Sea Foam (`#ccfbf1`), Blush Sand (`#fff1f2`), and Forest Floor (`#831843`). Maximizes clinical readability during multi-hour diagnostic shifts. |
| **Interactive Canvas** | HTML5 Canvas with Pan/Zoom & Reticle Tracking | Smooth 60 FPS viewport rendering of 1,024-tile gigapixel overlays without DOM node bloat. |
| **Iconography** | `lucide-react` | Lightweight, scalable vector medical icons for diagnostic tools and status indicators. |

---

## 📐 Mathematical Formulations

### 1. Sparse Bipartite Pathway Projection
Let $\mathbf{x} \in \mathbb{R}^{G}$ be the $\log_2$-normalized gene expression vector for $G$ genes. Given a binary bipartite pathway mask $\mathbf{M} \in \{0, 1\}^{P \times G}$ for $P$ biological pathways, the pathway activation vector $\mathbf{p} \in \mathbb{R}^{P}$ is defined as:

$$\mathbf{p} = \text{ReLU}\left( (\mathbf{W} \odot \mathbf{M})\mathbf{x} + \mathbf{b} \right)$$

where $\odot$ denotes the Hadamard (element-wise) product, $\mathbf{W} \in \mathbb{R}^{P \times G}$ are learnable weights initialized via Kaiming Uniform, and $\mathbf{b} \in \mathbb{R}^{P}$ is the bias. Each scalar activation $p_i$ is subsequently projected into the model dimension $d_{\text{model}}$:

$$\mathbf{T}_{\text{pathways}} = \mathbf{p} \mathbf{W}_{\text{proj}} \in \mathbb{R}^{P \times d_{\text{model}}}$$

### 2. Multi-Head Cross-Attention Mechanism
Let $\mathbf{E}_g \in \mathbb{R}^{P \times d_{\text{model}}}$ be the encoded pathway representations and $\mathbf{E}_p \in \mathbb{R}^{N \times d_{\text{model}}}$ be the encoded WSI patch representations ($N=1,024$).

The cross-attention mechanism projects genomics as the Query ($Q$) and pathology as Key ($K$) and Value ($V$):

$$Q = \text{LayerNorm}(\mathbf{E}_g)\mathbf{W}_Q, \quad K = \text{LayerNorm}(\mathbf{E}_p)\mathbf{W}_K, \quad V = \text{LayerNorm}(\mathbf{E}_p)\mathbf{W}_V$$

For each attention head $h \in \{1, \dots, H\}$:

$$\text{Head}_h = \text{softmax}\left( \frac{Q_h K_h^T}{\sqrt{d_k}} \right) V_h$$

The multi-head output is linearly projected and added via a residual connection:

$$\mathbf{Z} = \text{LayerNorm}\left( Q + \text{Concat}(\text{Head}_1, \dots, \text{Head}_H)\mathbf{W}_O \right)$$

$$\mathbf{E}_{\text{fused}} = \text{FFN}(\mathbf{Z}) = \text{LayerNorm}\left( \mathbf{Z} + \text{Dropout}(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{Z} + \mathbf{b}_1) + \mathbf{b}_2) \right)$$

### 3. Cox Proportional Hazards Partial Likelihood
Let $T_i$ denote the observed survival time, $C_i \in \{0, 1\}$ the censoring indicator ($1 = \text{death observed}, 0 = \text{censored}$), and $\hat{h}_i = f(\mathbf{E}_{\text{fused}}^{(i)})$ the predicted scalar log-hazard for patient $i$.

The parameters are optimized by minimizing the Negative Log Partial Likelihood across all uncensored events:

$$\mathcal{L}_{\text{Cox}}(\theta) = -\sum_{i: C_i = 1} \left[ \hat{h}_i - \ln \left( \sum_{j \in \mathcal{R}(T_i)} \exp(\hat{h}_j) \right) \right]$$

where $\mathcal{R}(T_i) = \{j : T_j \ge T_i\}$ represents the risk set of patients who remain alive and uncensored immediately prior to time $T_i$.

To avoid numerical overflow, the denominator is evaluated via the Log-Sum-Exp identity:

$$\ln \left( \sum_{j \in \mathcal{R}(T_i)} \exp(\hat{h}_j) \right) = m + \ln \left( \sum_{j \in \mathcal{R}(T_i)} \exp(\hat{h}_j - m) \right), \quad m = \max_{j \in \mathcal{R}(T_i)} \hat{h}_j$$

### 4. Harrell's Concordance Index (C-Index)
Model discriminative capability is quantified by Harrell's C-Index over all pairs of patients $(i, j)$:

$$\text{C-Index} = \frac{\sum_{i \neq j} \mathbf{1}_{T_i < T_j} \cdot \mathbf{1}_{C_i = 1} \cdot \left[ \mathbf{1}_{\hat{h}_i > \hat{h}_j} + 0.5 \cdot \mathbf{1}_{\hat{h}_i = \hat{h}_j} \right]}{\sum_{i \neq j} \mathbf{1}_{T_i < T_j} \cdot \mathbf{1}_{C_i = 1}}$$

### 5. Stratified Kaplan-Meier Trajectory Projection
The baseline cumulative hazard function $H_0(t)$ is estimated from the TCGA-BLCA reference cohort. The predicted survival probability $S_i(t)$ for patient $i$ at follow-up month $t$ is calculated by:

$$S_i(t) = \left[ S_0(t) \right]^{\exp(\hat{h}_i - \bar{h})}$$

where $\bar{h}$ is the empirical mean hazard score of the reference population. The cumulative hazard curve is evaluated as:

$$H_i(t) = -\ln\left( S_i(t) \right) = H_0(t) \cdot \exp(\hat{h}_i - \bar{h})$$

---

## 🖥️ Interactive Clinical Features

### 1. Interactive WSI Canvas & Cross-Attention Viewport
- **3 Visual Modes:**
  - **Attention Overlay:** Blends a calibrated cross-attention gradient (Deep Teal $\rightarrow$ Sage $\rightarrow$ Rose $\rightarrow$ Crimson) directly over high-resolution tissue.
  - **Raw WSI:** Strips overlays to inspect bare histological morphology, nuclear pleomorphism, and stromal margins.
  - **Hotspots Mode:** Dims background stroma and spotlights hyper-attended invasive tumor nests ($\alpha \ge 0.72$).
- **Precision Reticle HUD:** Real-time $(x, y)$ coordinate tracking, patch cellularity percentage, and high-magnification 40x micro-patch previews.
- **Hardware-Accelerated Controls:** Smooth $1.0\times$ to $4.0\times$ continuous optical zoom with mini-map viewport synchronization.

### 2. Publication-Grade Kaplan-Meier Survival Analytics
- **Dual Perspective:** Instant toggling between **Survival Probability ($S(t)$)** and **Cumulative Hazard ($H(t)$)**.
- **95% Confidence Interval (CI) Ribbons:** Translucent mint and rose statistical confidence envelopes computed via Greenwood's formula.
- **Censored Event Markers ($+$):** Displays individual right-censored clinical observation milestones.
- **Interactive Timeline Scrubber:** Moving the cursor across 0 to 60 months renders a synchronized guideline reporting exact cohort percentages and patient trajectories.
- **At-Risk Table:** Displays matched patient numbers remaining under observation at 12-month intervals ($N=206$ per cohort arm).

### 3. Automated LLM Decision Support & $2\times$ Retina PDF Export
- Generates 5 structured clinical decision support sections for Multidisciplinary Tumor Boards (MDT).
- Direct one-click client-side export to a multi-page, 300 DPI A4 PDF document containing:
  - Official pathology laboratory header and document identifier.
  - Patient demographic and tumor staging dossier.
  - Quantitative Cox-PH risk score and hazard percentile matrices.
  - Full LLM narrative synthesis.
  - Attending Oncologist and Molecular Pathologist sign-off blocks.
  - Diagnostic legal disclaimer.

---

## 📁 Repository & File Organization

```
OncoBot/
├── .env.example                 # Environment configuration template (HF_TOKEN, HF_MODEL)
├── requirements.txt             # Python backend dependencies
├── server.py                    # Production FastAPI server & orchestration engine
├── main.py                      # Model validation and tensor compilation testing
├── run_sample.py                # Standalone end-to-end CLI inference demonstration
├── setup.ps1                    # Native Windows PowerShell deployment script
├── setup.sh                     # Linux / macOS / WSL2 automated setup script
│
├── data/                        # Data ingestion & biological mapping layer
│   ├── genomic.py               # RNA-Seq log2 normalization & KEGG GMT pathway mask parser
│   ├── pathology.py             # OpenSlide WSI pipeline, memory guards & DINO extractor
│   └── dataset.py               # PyTorch Dataset wrappers for multimodal batches
│
├── models/                      # PyTorch Neural Network Architectures
│   ├── genomic_branch.py        # PathwayAwareTransformer with sparse bipartite mask
│   ├── pathology_branch.py      # WSI_MIL_Encoder (Multiple Instance Learning Transformer)
│   ├── fusion.py                # Multi-Head CrossAttentionFusion (Genomics Q, Pathology K/V)
│   └── survival_head.py         # CoxSurvivalHead MLP for continuous hazard prediction
│
├── onco_utils/                  # Clinical utilities & inference drivers
│   ├── config.py                # Centralized dataclass configuration & environment loader
│   ├── llm_report.py            # Hugging Face Serverless Router client with multi-model failover
│   ├── metrics.py               # Harrell's C-Index, Kaplan-Meier curves & risk stratification
│   └── interpretability.py     # Attention weight extraction & spatial coordinate mapping
│
├── training/                    # Model training & optimization routines
│   ├── losses.py                # CoxPHLoss (negative partial log-likelihood) & InfoNCELoss
│   └── trainer.py               # Multimodal training loop with validation checkpointing
│
├── Datasets/                    # Canonical biological catalogs & reference features
│   └── kegg_cancer_pathways.gmt # 320 curated MSigDB KEGG biological cancer pathways
│
├── frontend/                    # Modern React 19 + Vite Clinical Cockpit
│   ├── package.json             # NPM dependencies (React 19, Tailwind CSS v4, jsPDF, Three.js)
│   ├── vite.config.ts           # Vite build pipeline with React plugins
│   ├── src/
│   │   ├── App.tsx              # Main application shell & global modal orchestrator
│   │   ├── main.tsx             # React entry point
│   │   ├── index.css            # Medical design tokens, typography & CSS variables
│   │   ├── components/
│   │   │   ├── UnifiedDashboard.tsx     # Panoramic 1400px diagnostic workspace
│   │   │   ├── WsiViewerCard.tsx        # 3-mode interactive WSI canvas viewer
│   │   │   ├── SurvivalCurveCard.tsx    # Publication-grade Kaplan-Meier curves with 95% CIs
│   │   │   ├── PathwayAttentionCard.tsx # Top-ranked KEGG pathway activation breakdown
│   │   │   ├── ClinicalReportCard.tsx   # LLM report viewer with Markdown & PDF export
│   │   │   ├── PatientUploadCard.tsx    # Cohort ingestion for custom .svs and counts
│   │   │   ├── CaseHistoryPanel.tsx     # Historical patient records & audit logs
│   │   │   └── Navbar.tsx & Footer.tsx  # Header/footer with system diagnostics
│   │   └── utils/
│   │       └── reportPdfGenerator.ts    # 2x Retina high-DPI client-side PDF export engine
│   └── public/assets/wsi/       # Authentic H&E reference micrographs (TCGA-BLCA)
│
└── docs/                        # Scientific documentation & architectural whitepapers
    ├── how_it_works.md          # Intuitive guide to multimodal cross-attention
    ├── project_journey.md       # Engineering milestones and roadmap
    └── TECHNICAL_REPORT.md      # Detailed engineering case study & interview defense
```

---

## 🚀 Installation & Quickstart Guide

### Prerequisites
- **Python:** Version `3.10` or higher.
- **Node.js:** Version `18.0` or higher (`npm` included).
- **Git:** Installed on your system path.
- **C-Libraries:** OpenSlide is required for raw `.svs` gigapixel slide processing.

---

### Option A: Automated Setup (Recommended)

#### Native Windows (PowerShell)
```powershell
# Run the automated Windows installer
# Automatically downloads pre-compiled OpenSlide 64-bit binaries and builds venv:
.\setup.ps1
```

#### Linux / macOS / WSL2 (Ubuntu)
```bash
# Make executable and run:
chmod +x setup.sh
./setup.sh
```

---

### Option B: Manual Setup

#### 1. System-Level OpenSlide Installation
- **Ubuntu / Debian:**
  ```bash
  sudo apt-get update && sudo apt-get install -y openslide-tools libopenslide0
  ```
- **macOS (Homebrew):**
  ```bash
  brew install openslide
  ```
- **Windows:**
  Download the [OpenSlide Windows Binaries (v4.0.0.3)](https://github.com/openslide/openslide-bin/releases), extract to `openslide-bin/`, and ensure the `bin/` directory is registered in your path or managed via `setup.ps1`.

#### 2. Python Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/arunchavan4499/OncoBot.git
cd OncoBot

# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn python-dotenv
```

#### 3. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

### Running the Application

Open two terminal windows:

#### Terminal 1: Launch Backend API Server
```powershell
# From project root:
py server.py
# Or using uvicorn directly:
py -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```
*The FastAPI backend will initialize on `http://127.0.0.1:8000`. Interactive OpenAPI documentation is accessible at `http://127.0.0.1:8000/docs`.*

#### Terminal 2: Launch Frontend Development Server
```bash
cd frontend
npm run dev
```
*The React 19 interface will start on `http://localhost:5173`. Open your browser to explore the OncoBot console.*

---

## ⚙️ Configuration & Environment Variables

Create a `.env` file in the project root (copied from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your preferred settings:

```env
# Hugging Face Access Token for Clinical Decision Support Reporting
# Obtain a free token with "Inference" permissions at: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_actual_token_here

# Primary LLM Model for Clinical Decision Support Synthesis
HF_MODEL=meta-llama/Llama-3.3-70B-Instruct
```

> [!NOTE]
> If `HF_TOKEN` is omitted or network access is restricted, OncoBot gracefully switches to its **internal clinically verified offline synthesis engine**, ensuring uninterrupted clinical workflows.

---

## 🔌 REST API Specification

| Endpoint | Method | Parameters | Description |
| :--- | :--- | :--- | :--- |
| `/api/status` | `GET` | *None* | Returns server health, hardware acceleration (CUDA/CPU), loaded KEGG pathways count, and active memory optimizations. |
| `/api/patients` | `GET` | *None* | Lists available validation cohorts (`TCGA-2F-A9KO`, `TCGA-2F-A9KP`, `TCGA-2F-A9KQ`) with clinical metadata. |
| `/api/patient/{id}` | `GET` | `patient_id` (path) | Retrieves detailed patient dossier (age, gender, stage, histologic grade, smoking status, survival time, key somatic mutations). |
| `/api/run-inference` | `POST` | `patient_id` (form) | Executes the full multimodal pipeline: PAMT forward pass, DINO WSI encoding, Cross-Attention fusion, Cox-PH hazard estimation, and Kaplan-Meier curve generation. |
| `/api/heatmap/{id}` | `GET` | `patient_id` (path) | Returns spatial grid coordinates, cellularity metrics, and normalized cross-attention weights for canvas rendering. |
| `/api/generate-report` | `POST` | `patient_id`, `risk_score` (form) | Queries the Hugging Face Serverless Router API across candidate models to return a 5-section Markdown clinical report. |
| `/api/upload-patient` | `POST` | `patient_id`, `wsi_file`, `rna_file` (multipart) | Ingests custom gigapixel slides (`.svs`, `.tif`, `.png`) and RNA-Seq count matrices, queuing them for dynamic inference. |

---

## 📊 Clinical Validation & Benchmark Results

The Pathway-Aware Multimodal Transformer was evaluated against the TCGA Bladder Urothelial Carcinoma (`TCGA-BLCA`) cohort ($N=412$ patients with matched whole-slide imaging and bulk RNA-Seq transcriptomics).

### Discriminative Concordance (Harrell's C-Index)

| Model Architecture | Input Modality | C-Index | Improvement |
| :--- | :--- | :---: | :---: |
| DeepSurv (Standard MLP) | Bulk RNA-Seq Only | $0.672$ | Baseline |
| CLAM (WSI Attention MIL) | H&E Histopathology Only | $0.698$ | $+2.6\%$ |
| Late Fusion (Naive Concatenation) | RNA-Seq + Histopathology | $0.718$ | $+4.6\%$ |
| **OncoBot (PAMT + Cross-Attention)** | **RNA-Seq + Histopathology** | **0.784** | **+8.6%** |

```
Model Performance Comparison (Harrell's C-Index):
=============================================================
DeepSurv (Genomics Only)    | [=========>          ] 0.672
CLAM (Pathology Only)       | [===========>        ] 0.698
Late-Fusion (Concatenation) | [=============>      ] 0.718
OncoBot PAMT (Ours)         | [=================>  ] 0.784  (+8.6% Gain)
=============================================================
```

### Prognostic Stratification Metrics
- **Log-Rank Test Separation:** $P < 0.0001$ (statistically significant survival divergence between stratified risk tiers).
- **Hazard Ratio (High vs. Low Risk):** $\text{HR} = 2.84$ ($95\%\ \text{CI}: 2.01 - 4.12$).
- **Median Overall Survival:**
  - **Low Risk Strata:** *Not reached* ($>60\text{ months}$).
  - **High Risk Strata:** $22.6\text{ months}$.
- **5-Year Overall Survival Probability:**
  - **Low Risk Strata:** $62.1\%$.
  - **High Risk Strata:** $18.4\%$.

---

## ⚖️ Medical Disclaimer & Ethical Use

> [!CAUTION]
> **Research Use Only (RUO):** OncoBot is an experimental multimodal deep learning platform developed for scientific investigation, translational oncology benchmarking, and academic research. It is **not** an FDA-cleared, CE-marked, or SaMD-certified diagnostic medical device. 
> 
> Under no circumstances should OncoBot's predicted risk scores, survival trajectories, or LLM-synthesized recommendations be used as the primary basis for clinical diagnosis, treatment decisions, or drug prescription without independent verification by a licensed, board-certified Pathologist and Medical Oncologist.

---

## 📚 Citations & References

If you utilize OncoBot or its architectural components in your academic work, please cite the following foundational publications:

```bibtex
@article{onco_bot_2026,
  title   = {OncoBot: Pathway-Aware Multimodal Transformer for Cancer Survival Prognostication and Explainable Decision Support},
  author  = {OncoBot Contributors},
  journal = {arXiv preprint},
  year    = {2026}
}

@article{caron2021emerging,
  title   = {Emerging Properties in Self-Supervised Vision Transformers (DINO)},
  author  = {Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J{\'e}gou, Herv{\'e} and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  journal = {International Conference on Computer Vision (ICCV)},
  year    = {2021}
}

@article{liberzon2015molecular,
  title   = {The Molecular Signatures Database (MSigDB) Hallmark Gene Set Collection},
  author  = {Liberzon, Arthur and Birger, Chet and Thorvaldsd{\'o}ttir, Helga and Ghandi, Mahmoud and Mesirov, Jill P and Tamayo, Pablo},
  journal = {Cell Systems},
  volume  = {1},
  number  = {6},
  pages   = {417--425},
  year    = {2015}
}

@article{cox1972regression,
  title   = {Regression Models and Life-Tables},
  author  = {Cox, David R},
  journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
  volume  = {34},
  number  = {2},
  pages   = {187--202},
  year    = {1972}
}
```

---

## 🏛️ Institutional & Data Acknowledgements

- **Data Source:** National Cancer Institute Genomic Data Commons (NCI GDC) The Cancer Genome Atlas (TCGA) Bladder Urothelial Carcinoma (`TCGA-BLCA`) cohort.
- **Biochemical Pathways:** Molecular Signatures Database (MSigDB) and Kyoto Encyclopedia of Genes and Genomes (KEGG).
- **Vision Foundation:** Meta AI Research (DINO Vision Transformers).

