import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# Import custom modules
from data.genomic import GenomicDataLoader, PathwayMapper
from models.genomic_branch import PathwayAwareTransformer
from models.pathology_branch import WSI_MIL_Encoder
from models.fusion import CrossAttentionFusion
from models.survival_head import CoxSurvivalHead
from training.losses import InfoNCELoss, CoxPHLoss
from training.trainer import MultimodalTrainer

class OncoMultiModalDataset(Dataset):
    """
    Dataset that aligns the WSI features, genomic features, and survival data.
    Gracefully handles cases where data might be missing.
    """
    def __init__(self, csv_path, tsv_path, survival_path, wsi_features_dir):
        self.df = pd.read_csv(csv_path)
        self.surv_df = pd.read_csv(survival_path, sep='\t')
        
        # Build mapping for survival
        self.surv_map = self.surv_df.drop_duplicates(subset=['_PATIENT']).set_index('_PATIENT')[['OS.time', 'OS']].to_dict('index')
        
        self.genomic_loader = GenomicDataLoader(tsv_path)
        self.wsi_features_dir = wsi_features_dir
        
        self.data = []
        for _, row in self.df.iterrows():
            pid = row['case_barcode']
            
            # 1. Check if patient has pre-extracted .pt features
            pt_path = os.path.join(self.wsi_features_dir, f"{pid}_features.pt")
            if not os.path.exists(pt_path):
                continue
                
            # 2. Check genomic data
            g_feats = self.genomic_loader.get_patient_data(pid)
            if g_feats is None:
                continue
                
            # 3. Check survival data
            if pid not in self.surv_map:
                continue
                
            surv_info = self.surv_map[pid]
            
            self.data.append({
                'patient_id': pid,
                'genomic_features': g_feats.values,
                'wsi_path': row.get('aws_url', ''), # Passed to trainer for fallback if needed
                'survival_time': surv_info['OS.time'],
                'censor': surv_info['OS']
            })
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Hyperparameters
    d_model = 128
    batch_size = 4  # Small batch to prevent OOM
    epochs = 10
    lr = 1e-4
    
    csv_path = 'Datasets/aligned_3way_slides_MINIMAL_case_level.csv'
    tsv_path = 'Datasets/TCGA-BLCA.star_counts.tsv'
    survival_path = 'Datasets/TCGA-BLCA.survival.tsv'
    wsi_features_dir = 'Datasets/WSI_Features'
    gmt_path = 'Datasets/kegg_cancer_pathways.gmt'
    
    # 2. Data Loading
    print("Loading datasets...")
    dataset = OncoMultiModalDataset(csv_path, tsv_path, survival_path, wsi_features_dir)
    print(f"Loaded {len(dataset)} aligned patient records.")
    
    if len(dataset) == 0:
        print("No valid data found. Exiting.")
        return

    # Train/Test Split (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    print(f"Split data into {train_size} training samples and {test_size} testing samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_genes = len(dataset.genomic_loader.data)
    gene_list = dataset.genomic_loader.data.index.tolist()
    
    # Use real KEGG pathways
    mapper = PathwayMapper(gene_list, gmt_path=gmt_path)
    num_pathways = mapper.num_pathways
    pathway_mask = mapper.pathway_mask.to(device)

    # 3. Model Initialization
    print("Initializing models...")
    genomic_model = PathwayAwareTransformer(
        num_genes=num_genes, 
        num_pathways=num_pathways, 
        pathway_mask=pathway_mask,
        d_model=d_model
    )
    
    pathology_model = WSI_MIL_Encoder(
        input_dim=384, # DINO vits16 dim
        d_model=d_model,
        num_patches=1024
    )
    
    fusion_model = CrossAttentionFusion(d_model=d_model)
    survival_head = CoxSurvivalHead(input_dim=d_model)
    
    # 4. Loss & Optimizer
    info_nce_loss = InfoNCELoss(temperature=0.1)
    cox_loss = CoxPHLoss()
    
    params = list(genomic_model.parameters()) + \
             list(pathology_model.parameters()) + \
             list(fusion_model.parameters()) + \
             list(survival_head.parameters())
             
    optimizer = optim.Adam(params, lr=lr)
    
    # 5. Trainer
    trainer = MultimodalTrainer(
        genomic_model=genomic_model,
        pathology_model=pathology_model,
        fusion_model=fusion_model,
        survival_head=survival_head,
        info_nce_loss_fn=info_nce_loss,
        cox_loss_fn=cox_loss,
        optimizer=optimizer,
        device=device,
        features_output_dir=wsi_features_dir
    )
    
    # 6. Training Loop
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, alpha=0.5)
        val_loss = trainer.evaluate_epoch(test_loader, alpha=0.5)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # 7. Save Model Weights
    print("Saving trained model weights...")
    os.makedirs("weights", exist_ok=True)
    torch.save(genomic_model.state_dict(), "weights/genomic_model.pth")
    torch.save(pathology_model.state_dict(), "weights/pathology_model.pth")
    torch.save(fusion_model.state_dict(), "weights/fusion_model.pth")
    torch.save(survival_head.state_dict(), "weights/survival_head.pth")
    print("Weights saved successfully to weights/ directory!")

if __name__ == '__main__':
    main()
