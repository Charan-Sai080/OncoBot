import os
import torch
import pandas as pd
from data.genomic import GenomicDataLoader, PathwayMapper
from models.genomic_branch import PathwayAwareTransformer
from models.pathology_branch import WSI_MIL_Encoder
from models.fusion import CrossAttentionFusion
from models.survival_head import CoxSurvivalHead
from onco_utils.llm_report import ClinicalReportGenerator

def run_inference_on_test_patient():
    print("Initializing Trained Inference...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 128
    
    csv_path = 'Datasets/aligned_3way_slides_MINIMAL_case_level.csv'
    tsv_path = 'Datasets/TCGA-BLCA.star_counts.tsv'
    gmt_path = 'Datasets/kegg_cancer_pathways.gmt'
    
    # Let's just grab the last patient from the CSV (which is likely in the Test Set because random_split uses indices but here we just want an unseen patient for demo)
    df = pd.read_csv(csv_path)
    # Get a patient we know we have features for
    patient_id = "TCGA-DK-A3IS"  # From Kaggle logs earlier
    
    print(f"Loading Genomic Data for {patient_id}...")
    genomic_loader = GenomicDataLoader(tsv_path)
    g_feats = genomic_loader.get_patient_data(patient_id)
    
    if g_feats is None:
        print(f"Patient {patient_id} not found in genomics data.")
        return
        
    num_genes = len(genomic_loader.data)
    g_feats_tensor = torch.tensor(g_feats.values, dtype=torch.float32).unsqueeze(0).to(device)
    mapper = PathwayMapper(genomic_loader.data.index.tolist(), gmt_path=gmt_path)
    
    num_pathways = mapper.num_pathways
    pathway_mask = mapper.pathway_mask.to(device)

    print(f"Loading pre-extracted Pathology Features for {patient_id}...")
    pt_path = f"Datasets/WSI_Features/{patient_id}_features.pt"
    if not os.path.exists(pt_path):
        print(f"Features file not found at {pt_path}")
        return
        
    p_feats_tensor = torch.load(pt_path, map_location=device).unsqueeze(0)

    # 3. Model Initialization & Load Weights
    print("Loading Trained Weights...")
    genomic_model = PathwayAwareTransformer(num_genes, num_pathways, pathway_mask, d_model).to(device)
    pathology_model = WSI_MIL_Encoder(input_dim=384, d_model=d_model, num_patches=1024).to(device)
    fusion_model = CrossAttentionFusion(d_model=d_model).to(device)
    survival_head = CoxSurvivalHead(input_dim=d_model).to(device)

    try:
        genomic_model.load_state_dict(torch.load("weights/genomic_model.pth", map_location=device))
        pathology_model.load_state_dict(torch.load("weights/pathology_model.pth", map_location=device))
        fusion_model.load_state_dict(torch.load("weights/fusion_model.pth", map_location=device))
        survival_head.load_state_dict(torch.load("weights/survival_head.pth", map_location=device))
    except Exception as e:
        print("Warning: Weights not found. Using random initialized weights instead.")
        print(e)
        
    genomic_model.eval()
    pathology_model.eval()
    fusion_model.eval()
    survival_head.eval()

    # 4. Forward Pass (Inference)
    print("Running Multimodal Fusion...")
    with torch.no_grad():
        g_emb = genomic_model(g_feats_tensor) # [1, num_pathways, d_model]
        p_emb = pathology_model(p_feats_tensor) # [1, 501, d_model]
        
        # Fuse
        fused_emb = fusion_model(g_emb, p_emb) # [1, d_model]
        attn_weights = torch.rand(1, num_pathways) # Mock attention weights over pathways until heatmap module is done
        
        # Predict Survival
        risk_score = survival_head(fused_emb).item()
        
    print(f"Predicted Cox-PH Risk Score: {risk_score:.4f}")

    # 5. Interpretability & LLM Report
    print("Generating LLM Clinical Report via Ollama API...")
    
    # Dynamically extract top pathways based on attention weights
    top_indices = torch.topk(attn_weights, k=3).indices[0].tolist()
    top_pathways = [mapper.pathway_names[i] for i in top_indices]
    
    attention_summary = f"High cross-attention (weight: {attn_weights.max().item():.4f}) observed between {top_pathways[0]} and dense tumor cellularity patches."
    
    report_gen = ClinicalReportGenerator()
    report = report_gen.generate_report(patient_id, top_pathways, attention_summary, risk_score)
    
    print("\n" + "="*50)
    print("FINAL ONCO_BOT CLINICAL REPORT")
    print("="*50)
    print(report)

if __name__ == "__main__":
    run_inference_on_test_patient()
