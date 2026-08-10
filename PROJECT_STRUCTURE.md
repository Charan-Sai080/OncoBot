# Onco_Bot 🧬 - In-Depth Project Structure

This document provides a detailed breakdown of the exact folder structure and the internal logic of every single file in the Onco_Bot repository. Use this to orient new contributors or as a reference during your final year presentation.

---

## 📂 Root Directory
The root directory holds the main entry points for the application, deployment scripts, and overarching documentation.

* **`main.py`**
  * **What it does:** This is the core architectural blueprint and model testing script. It defines the hyperparameters (like `d_model`, `num_heads`) and instantiates the raw PyTorch models to ensure they compile and can pass dummy tensors through the network without shape mismatches.
* **`run_sample.py`**
  * **What it does:** This is the **End-to-End Inference Engine**. When you execute this script, it orchestrates the entire pipeline: it triggers the data loaders to extract WSI patches, passes the features through the neural networks, executes the multimodal fusion, calculates the final Cox-PH risk score, and invokes the LLM API.
* **`setup.sh`** & **`setup.ps1`**
  * **What they do:** These are universal deployment scripts. `setup.sh` automatically detects if the user is on macOS or Linux and installs the notoriously difficult `openslide` C-libraries via `brew` or `apt-get`. `setup.ps1` provides native Windows support by dynamically downloading pre-compiled OpenSlide `.dll` binaries. Both scripts automatically build your Python Conda/Venv environment.
* **`requirements.txt`**
  * **What it does:** A standard Python dependency file listing all required packages (PyTorch, OpenSlide, OpenCV, etc.) for non-Conda users.
* **`TECHNICAL_REPORT.md`** & **`TEAMMATE_HANDOVER.md`**
  * **What they do:** Deep-dive case studies explaining the engineering decisions, memory optimizations, and handover instructions for UI contributors.

---

## 📂 `data/` (The Data Ingestion Layer)
This folder handles the heavy lifting of parsing gigabytes of raw biological data and transforming it into lightweight tensors.

* **`data/genomic.py`**
  * **What it does:** Processes `.csv` files containing Bulk Transcriptomics (RNA-Seq). It normalizes raw gene expressions and, most importantly, constructs the **Boolean Pathway Mask**. This mask maps individual genes to specific biological pathways (like p53 or PI3K-Akt) so the model learns biologically grounded features rather than arbitrary mathematical noise.
* **`data/pathology.py`**
  * **What it does:** The most computationally complex file in the project. It connects to the `OpenSlide` C-engine to read massive 4GB+ `.svs` Whole Slide Images. 
  * **Key Optimizations inside:** It generates a 2048x2048 low-res thumbnail to identify where tissue exists (bypassing a 40GB RAM allocation bug). It then uses **Controlled Multiprocessing** to spawn 4 workers that randomly extract exactly 1,024 patches (the "sweet spot") while forcefully closing the C-handles after every chunk to prevent aggressive memory caching and SSD swap wear-out. Finally, it passes those patches through a frozen **DINO Vision Transformer** to extract `384-dimensional` visual embeddings.

---

## 📂 `models/` (The Neural Architecture Layer)
This folder contains the actual PyTorch Neural Networks that learn from the data.

* **`models/genomic_branch.py`**
  * **What it does:** Contains the **Pathway-Aware Transformer**. Unlike a standard Multi-Layer Perceptron (MLP) that is a "black box", this model applies the boolean pathway mask from the data layer during its forward pass. This strictly forces the network to calculate attention weights *between* distinct biological pathways, ensuring high interpretability.
* **`models/pathology_branch.py`**
  * **What it does:** Contains the **WSI MIL (Multiple Instance Learning) Encoder**. It takes the 1,024 unordered DINO visual patch embeddings and passes them through a Transformer Encoder. This allows the model to learn the global context of the tumor microenvironment by analyzing how different tissue patches relate to one another spatially.
* **`models/fusion.py`**
  * **What it does:** The heart of the multimodal system. It houses the **Cross-Attention Fusion** module. It takes the output from the Genomic Branch (acting as the *Query*) and the Pathology Branch (acting as the *Key/Value*). This allows the genes to "look at" specific tissue regions. It outputs the fused representation to a **Cox-PH Survival Head**, which calculates the final patient risk score.

---

## 📂 `onco_utils/` (The Utilities & Cloud Layer)
This folder contains helper scripts that bridge the gap between the AI math and the end user.

* **`onco_utils/llm_report.py`**
  * **What it does:** Contains the `ClinicalReportGenerator` class. It takes the raw numerical outputs from the PyTorch model (the Cox-PH risk score and the highly-attended biological pathways) and formats them into a rigorous text prompt. It securely authenticates with the **Ollama Cloud API** (e.g., using `minimax-m3` or Qwen) to generate the final human-readable executive clinical summary.
* **`onco_utils/heatmap.py`** *(To be built by UI Contributor)*
  * **What it does:** Will intercept the Cross-Attention weights from the fusion model and use OpenCV to spatially map a color-coded heatmap back onto the original WSI image, visually explaining the model's predictions.
