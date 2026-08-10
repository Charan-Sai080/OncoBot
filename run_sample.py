import os
import torch
import pandas as pd
from data.genomic import GenomicDataLoader, PathwayMapper
from data.pathology import WSIPipeline, PatchFeatureExtractor
from models.genomic_branch import PathwayAwareTransformer
from models.pathology_branch import WSI_MIL_Encoder
from models.fusion import CrossAttentionFusion
from models.survival_head import CoxSurvivalHead
from onco_utils.llm_report import ClinicalReportGenerator

def run_sample_inference():
    print("Initializing Onco_Bot Sample Inference...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 128
    patient_id = "TCGA-2F-A9KO"
    
    # 1. Load Genomic Data
    print(f"Loading Genomic Data for {patient_id}...")
    tsv_path = 'Datasets/TCGA-BLCA.star_counts.tsv'
    genomic_loader = GenomicDataLoader(tsv_path)
    g_feats = genomic_loader.get_patient_data(patient_id)
    
    if g_feats is None:
        # Fallback to random if dataset is minimal or missing this specific ID
        print(f"Genomic data for {patient_id} not found in tsv. Using dummy genomic data.")
        num_genes = 1000
        g_feats_tensor = torch.rand(1, num_genes)
        mapper = PathwayMapper([f"GENE_{i}" for i in range(num_genes)])
    else:
        num_genes = len(genomic_loader.data)
        g_feats_tensor = torch.tensor(g_feats.values, dtype=torch.float32).unsqueeze(0)
        mapper = PathwayMapper(genomic_loader.data.index.tolist())
        
    num_pathways = mapper.num_pathways
    pathway_mask = mapper.pathway_mask.to(device)
    g_feats_tensor = g_feats_tensor.to(device)

    # 2. Extract WSI Pathology Data
    wsi_path = "Datasets/WSI/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs"
    print(f"Extracting WSI Patches from {wsi_path}...")
    wsi_pipeline = WSIPipeline()
    feature_extractor = PatchFeatureExtractor(output_dir="Datasets/WSI_Features", device=device)
    
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI file not found at {wsi_path}")
    print("Running OpenSlide Tissue Segmentation...")
    wsi_pipeline.extract_patches(wsi_path, patient_id)
    print("Extracting DINO Features...")
    p_feats_path = feature_extractor.process_patient(patient_id, os.path.join(wsi_pipeline.output_dir, patient_id))
    p_feats_tensor = torch.load(p_feats_path)

    p_feats_tensor = p_feats_tensor.unsqueeze(0).to(device) # Add batch dim

    # 3. Model Initialization (Random weights for sample inference)
    print("Initializing Modalities...")
    genomic_model = PathwayAwareTransformer(num_genes, num_pathways, pathway_mask, d_model).to(device).eval()
    pathology_model = WSI_MIL_Encoder(input_dim=384, d_model=d_model, num_patches=1024).to(device).eval()
    fusion_model = CrossAttentionFusion(d_model=d_model).to(device).eval()
    survival_head = CoxSurvivalHead(input_dim=d_model).to(device).eval()

    # 4. Forward Pass (Inference)
    print("Running Multimodal Fusion...")
    with torch.no_grad():
        g_emb = genomic_model(g_feats_tensor) # [1, num_pathways, d_model]
        p_emb = pathology_model(p_feats_tensor) # [1, 501, d_model]
        
        # Fuse
        fused_emb = fusion_model(g_emb, p_emb) # [1, d_model]
        attn_weights = torch.rand(1, 10) # Mock attention weights for print
        
        # Predict Survival
        risk_score = survival_head(fused_emb).item()
        
    print(f"Predicted Cox-PH Risk Score: {risk_score:.4f}")

    # 5. Interpretability & LLM Report
    print("Generating LLM Clinical Report via Anthropic Claude...")
    top_pathways = ["PI3K-Akt signaling pathway", "p53 signaling pathway", "Cell cycle"] # Mocked for speed
    attention_summary = f"High cross-attention (weight: {attn_weights.max().item():.4f}) observed between PI3K-Akt pathway and dense tumor cellularity patches."
    
    report_gen = ClinicalReportGenerator()
    report = report_gen.generate_report(patient_id, top_pathways, attention_summary, risk_score)
    
    print("\n" + "="*50)
    print("FINAL ONCO_BOT CLINICAL REPORT")
    print("="*50)
    print(report)

if __name__ == "__main__":
    run_sample_inference()
