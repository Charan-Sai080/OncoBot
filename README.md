# Onco_Bot 🧬🔬

**Onco_Bot** is a multimodal AI pipeline designed to integrate Whole-Slide Histopathology (WSI) and Bulk Transcriptomics (Genomics) to predict patient survival risk (Cox-PH) and generate comprehensive human-readable clinical reports via Large Language Models.

This project was built to run efficiently on local hardware (like Apple Silicon Macs or Linux NAS systems) by utilizing aggressive memory management and controlled multiprocessing to avoid SSD wear-out during heavy WSI extraction.

## Features
* **Multimodal Fusion Engine:** Cross-attends DINO visual embeddings with a Pathway-Aware Transformer for transcriptomics.
* **OpenSlide Memory Optimization:** Prevents massive C-level memory leaks by forcing thumbnail masking and explicitly destroying memory handles across chunked multiprocessing workers.
* **Sweet-Spot Sampling:** Randomly samples exactly 1,024 patches (bypassing K-Means bottlenecks) to map the tumor microenvironment quickly.
* **Ollama Cloud API Integration:** Automatically translates risk scores and cross-attention weights into an executive clinical summary using commercial LLM endpoints.

---

## 🚀 Quick Start Guide

### 1. Universal Setup Script
For new contributors or deploying to a new system (Mac or Linux), simply run the included setup script. This script automatically detects your OS, installs the required system-level C libraries (like `openslide`), and configures your Python environment.

**Mac / Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Native Windows (PowerShell):**
```powershell
.\setup.ps1
```

### 2. Manual Installation (Optional)
If you prefer not to use the automated script, you can install the dependencies manually:

**System Requirements:**
OpenSlide is required to process `.svs` files.
* **macOS:** `brew install openslide`
* **Linux (Debian/Ubuntu):** `sudo apt-get install openslide-tools libopenslide0`
* **Windows (NVIDIA GPU):** Native Windows is not recommended due to OpenSlide C-library compilation issues. **We strongly recommend installing WSL2 (Ubuntu)** on Windows. Once inside WSL2, PyTorch will automatically detect your NVIDIA GPU for full CUDA acceleration. Simply run the Linux `apt-get` command above inside your WSL terminal.

**Python Requirements:**
If you are using Conda:
```bash
conda create -n onco_bot python=3.10 -y
conda activate onco_bot
pip install -r requirements.txt
```
If you are using a standard Virtual Environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Running Inference
To run a sample end-to-end inference pass on a test patient (`TCGA-2F-A9KO`):

```bash
# If using Conda:
conda activate onco_bot

# If using Venv:
source venv/bin/activate

# Execute the pipeline:
python run_sample.py
```

*Note: On some macOS systems, you may need to prepend `KMP_DUPLICATE_LIB_OK=TRUE` to the python command if you encounter OpenMP library conflicts.*

---

## 🛠️ Project Structure
* `data/pathology.py` - Advanced WSI patch extraction with Multiprocessing and Memory optimizations.
* `models/` - PyTorch architectures for the Pathway-Aware Transformer, WSI MIL Encoder, and Cross-Attention Fusion.
* `onco_utils/` - Utility scripts, including the `llm_report.py` module for hitting the Ollama Cloud API.
* `run_sample.py` - The main entry point to test the entire E2E inference pipeline.
