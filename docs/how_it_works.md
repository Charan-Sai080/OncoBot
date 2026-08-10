# How Onco_Bot Works: A Simple & Technical Guide

This document explains the core concepts behind the Onco_Bot multimodal transformer in a way that bridges complex deep learning with straightforward logic.

## The Core Problem
In cancer survival prediction, doctors usually look at two things:
1.  **Genetics (RNA-seq)**: The "code" or "instructions" of the cancer cells.
2.  **Histopathology (Images)**: The physical appearance of the tumor under a microscope.

Historically, AI models either looked at just the genes, or just the images. Even when they looked at both, they just clumsily smashed the data together (e.g., concatenated a list of numbers). 
**Onco_Bot's goal is to understand *how* the genetic code physically manifests in the tissue.**

## Step 1: Making Sense of Genes (The Pathway Approach)
Humans have ~60,000 genes. Throwing 60,000 random numbers at an AI is messy. 
Instead of looking at individual genes, Onco_Bot groups them into **Biological Pathways** (like a "DNA repair team" or an "immune response team"). 
*   **Technical translation**: We map genes to 186 KEGG pathways. The AI learns how these 186 "teams" interact using a Transformer (the same tech behind ChatGPT).

## Step 2: Making Sense of Massive Images (The Patch Approach)
A medical image of a tumor (Whole-Slide Image) is too massive to feed into an AI all at once. 
*   **Technical translation**: We chop the giant image into thousands of tiny squares (patches). Since many squares look identical (e.g., normal healthy tissue), we cluster them and pick the 500 most unique, representative squares. We use a visual AI (DINO) to convert these 500 squares into mathematical summaries (embeddings).

## Step 3: The Magic (Cross-Attention)
This is the heart of the paper. We need to fuse the 186 Gene Pathways with the 500 Image Patches.
We do this using **Cross-Attention**. Think of it like this:
The AI takes a specific genetic pathway (e.g., "Immune Response") and acts as a **Query** (a searchlight). It shines this searchlight across all 500 image patches to see which patches light up (which patches physically show signs of an immune response).
*   **Technical translation**: The Pathways act as the Query (Q), and the Image Patches act as the Key (K) and Value (V) in the Transformer block. The AI mathematically learns the correlation between genotype (genes) and phenotype (images).

## Step 4: Predicting Survival
Once the AI has mapped how the genes and images interact, it crunches that fused information down to a single number: a **Survival Risk Score**. It compares this score against actual patient data during training to learn how to be accurate.

## Step 5: The "Why" (Explainability)
Because we used "Cross-Attention" (the searchlight), the AI leaves a paper trail. 
We can literally look at the math to see *where* the searchlight was pointing. 
*   **Heatmaps**: We can overlay colors on the tumor image. "Red" areas mean "The AI was looking exactly here because of a specific genetic mutation."
*   **The Novelty (LLM Report)**: We take this paper trail and hand it to a Language Model. The Language Model translates the math into a clinical summary: *"This patient is high risk because Pathway X was highly active and correlated with abnormal cell growth in patches A, B, and C."*
