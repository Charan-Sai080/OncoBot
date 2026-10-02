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

## 🔬 Clinical Histopathology Viewer & WSI Upload Pipeline

Onco_Bot includes an interactive, multi-layer histopathology viewport that connects whole-slide imaging with genomic pathway cross-attention.

### 1. Pre-configured Demo Cases (Instant Testing)
* For the bundled demo cohort (`TCGA-2F-A9KO`, `TCGA-2F-A9KP`, `TCGA-2F-A9KQ`), the dashboard renders high-resolution, biologically authentic **H&E histopathology micrographs** modeling human urothelial bladder carcinoma (located in `frontend/public/assets/wsi/`).
* Allows testing of the entire multimodal interface—including real-time spatial zoom, reticle tracking, tile classification, and LLM clinical reporting—without requiring immediate multi-gigabyte slide transfers.

### 2. Uploading Real Patient Images
* **Supported Formats:** `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, and whole-slide `.svs` files (or direct Google Drive cloud links).
* **End-to-End Workflow:**
  1. Click **"Add Patient"** on the home dashboard to expand the cohort ingestion form.
  2. Upload your real tumor histology image or biopsy crop under **Whole-Slide Histopathology** (alongside optional RNA-Seq count matrices).
  3. The FastAPI backend automatically captures the file, persists it to the `uploads/` directory, and registers the custom patient case.
  4. The model tiles the real tissue, runs cross-attention with the genomic embeddings, and generates spatial attention matrices.
  5. The interactive canvas viewer dynamically decodes and renders your **actual uploaded cancer tissue**, overlaying predictive risk heatmaps directly aligned with your specimen morphology.

### 3. Interactive Canvas Viewing Modes
* **Attention Overlay Mode:** Blends a calibrated cross-attention colormap (Deep Teal $\rightarrow$ Sage $\rightarrow$ Rose $\rightarrow$ Crimson) over the real cancer tissue to reveal regions driving the survival risk score.
* **Raw WSI Mode:** Strips off the attention heatmap to inspect the raw, unmasked tissue histology, nuclear pleomorphism, and stromal architecture with subtle grid boundaries.
* **Hotspots Mode:** Dims background tissue and spotlights high-attention clusters ($\ge 0.72$) representing critical invasive tumor nests and lymphocyte infiltration zones.
* **Inspection Dock:** Provides up to 4.0x zoom, macro slide viewfinder tracking, coordinate locator $(x, y)$, cellular density metrics, and high-magnification 40x micro-patch previews.

---

## 📈 Stratified Kaplan-Meier Survival Curves & Prognostic Modeling

The prognostic module provides publication-grade, interactive time-to-event survival analysis, projecting the patient's personalized trajectory against benchmarked clinical cohorts from TCGA-BLCA.

### 1. Dual-View Mode
* **Survival Probability ($S(t)$):** Plots overall survival probability from 100% down across 60 months, highlighting divergence between risk tiers.
* **Cumulative Hazard ($H(t)$):** Toggles to cumulative hazard mode ($H(t) = -\ln(S(t))$) to visualize the cumulative event rate over the follow-up timeline.

### 2. Publication-Quality Curve Features
* **95% Confidence Interval (CI) Ribbons:** Smooth translucent ribbons (Pastel Mint for Low Risk, Pastel Rose for High Risk) visualizing statistical confidence bounds.
* **Censored Event Markers ($+$):** Displays individual right-censored patient events along both cohort curves.
* **Patient Trajectory Nodes:** Distinct deep blue-teal curve featuring hollow circular tracking nodes along evaluation intervals.
* **Interactive Timeline Scrubber:** Moving the cursor across the chart renders a vertical guideline with colored intersection nodes and a floating HUD tooltip reporting exact cohort percentages and confidence intervals at any month.

### 3. Key Clinical Statistics & At-Risk Table
* **Statistical Metrics:**
  * **Log-Rank Test:** $P < 0.0001$ (statistically significant cohort separation).
  * **Hazard Ratio (HR):** $2.84$ ($95\%\ \text{CI}: 2.01 - 4.12$).
  * **Median Survival:** Low Risk (*Not reached*) vs. High Risk (*22.6 months*).
  * **5-Year Overall Survival:** Low Risk (*62.1%*) vs. High Risk (*18.4%*).
* **At-Risk Table:**
  * Displays matched patient numbers at risk at $0, 12, 24, 36, 48,$ and $60$ months ($N=206$ per cohort arm).

---

## 🖥️ Running the Web Application

The system consists of a FastAPI backend and a modern React + Vite clinical dashboard.

### 1. Start the Backend API Server
```powershell
# From the project root:
py server.py
# Or with hot-reload:
py -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```
*API docs and interactive OpenAPI specifications are available at `http://127.0.0.1:8000/docs`.*

### 2. Start the Frontend Development Server
```bash
cd frontend
npm install
npm run dev
```
*Open `http://localhost:5173` in your browser to launch the Onco_Bot Clinical AI dashboard.*
