# Onco_Bot: Newbie Onboarding Guide & Architecture Summary

Welcome to the **Onco_Bot** project! If you are a new teammate joining this project, this document will explain exactly what this system is, how it works, and the engineering problems we solved to build it. 

Do not worry if you are new to AI—this guide breaks down the complex math into simple concepts.

---

## 1. What is Onco_Bot?
Onco_Bot is a **Multimodal Artificial Intelligence** pipeline for Bladder Cancer. 
When a patient gets cancer, doctors look at two things:
1. **Histopathology (WSI):** A massive gigapixel image of the tumor tissue (how it *looks*).
2. **Genomics (RNA-Seq):** The patient's DNA/RNA data (how it *behaves* biologically).

Normally, AI only looks at one of these. Onco_Bot looks at **both simultaneously**, fusing them together to accurately predict how long the patient will survive (the Cox-PH Survival Risk Score). It then uses a Large Language Model (LLM) to write a clinical report for the doctor explaining *why* it made that prediction.

---

## 2. How the Pipeline Works (Step-by-Step)

### Step 1: The Vision Pipeline (Handling the Giant Images)
Whole-Slide Images (WSIs) are around 4GB *each*. If we fed them directly into a standard neural network, any normal computer would instantly crash.
* **What we achieved:** We built a smart patch-extraction system using Python's `openslide`. It safely scans the 4GB image, finds the actual tumor tissue, and extracts exactly 1,024 small "patches". 
* **The AI Model:** We push these 1,024 patches through an off-the-shelf Vision Transformer called **DINO** (created by Meta). DINO translates the visual pixels into a dense mathematical matrix (a 384-dimensional tensor).

### Step 2: The Genomic Pipeline (Handling 20,000 Genes)
Humans have about 20,000 genes. If we feed all 20,000 into a small neural network, the AI will overfit and memorize the data.
* **What we achieved:** We downloaded the official **KEGG Pathways database** (from the Broad Institute). This groups thousands of genes into 320 known biological systems (e.g., the p53 tumor-suppressor pathway).
* **The AI Model:** We built a custom `PathwayAwareTransformer`. It uses the KEGG database as a "mask" to physically force the neural network to only evaluate genes that belong to the same biological pathway.

### Step 3: Multimodal Fusion (Where the Magic Happens)
Now we have the math for the Image, and the math for the Genes. How do we combine them?
* **What we achieved:** We implemented a **Cross-Attention Mechanism** (`models/fusion.py`). 
* **How it works:** The AI uses the Genomic Pathways as a "Query", and searches through the 1,024 Image patches to find matches. It literally calculates the mathematical relationship between a specific gene pathway and a specific physical region of the tumor!

### Step 4: Survival Prediction & LLM Reporting
* The fused data is passed into a **Cox-PH Survival Head**, which calculates the final Risk Score (e.g., 0.0294).
* A Python script intercepts the top 3 biological pathways the AI focused on during Step 3, and sends them to the **Ollama Cloud LLM API**. The LLM writes a professional medical report for the doctor.

---

## 3. How We Are Training It (The Distributed Strategy)

As a team, you do not need supercomputers to train this! We are using a clever **Distributed MLOps** strategy:

1. **Google Colab Pro (The Heavy Lifting):** We upload the heavy 4GB images to Google Drive. Using Colab's powerful GPUs, we run the DINO Vision model to extract the mathematical features and save them as tiny 1.5MB `.pt` files.
2. **Local Mac/PC (The Smart Training):** We download those tiny `.pt` files to our local computers. Because the files are so small, we can train the custom Genomic Transformer and the Cross-Attention Fusion model directly on our laptops without crashing our hard drives!

---

## 4. Your Next Steps as a Teammate
To run this pipeline yourself and see the magic happen:
1. Ensure your conda environment is activated (`conda activate onco_bot`).
2. Run the main inference script:
   ```bash
   KMP_DUPLICATE_LIB_OK=TRUE python run_sample.py
   ```
3. Watch the terminal as it extracts the image patches, calculates the KEGG pathways, predicts the survival score, and streams the LLM medical report!
