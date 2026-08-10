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

class DummyMultiModalDataset(Dataset):
    """
    Temporary dataset wrapper. 
    In production, this aligns the 'aligned_3way_slides_MINIMAL_case_level.csv' 
    and 'TCGA-BLCA.star_counts.tsv' correctly.
    """
    def __init__(self, csv_path, tsv_path):
        self.df = pd.read_csv(csv_path)
        self.genomic_loader = GenomicDataLoader(tsv_path)
        
        # Simple alignment matching patient ID (case_id)
        self.data = []
        for _, row in self.df.iterrows():
            pid = row.get('case_id', row.iloc[0])
            g_feats = self.genomic_loader.get_patient_data(pid)
            if g_feats is not None:
                self.data.append({
                    'patient_id': pid,
                    'genomic_features': g_feats.values,
                    'wsi_path': row['slide_path'] if 'slide_path' in row else row.get('s3_url', 'file:///tmp/dummy.svs'),
                    'survival_time': row.get('survival_time', np.random.uniform(10, 100)),
                    'censor': row.get('censor', np.random.choice([0, 1]))
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
    
    csv_path = 'data/aligned_3way_slides_MINIMAL_case_level.csv'
    tsv_path = 'data/TCGA-BLCA.star_counts.tsv'
    
    # Check if data exists, else skip for demonstration
    if not os.path.exists(csv_path) or not os.path.exists(tsv_path):
        print(f"Data files missing. Please ensure {csv_path} and {tsv_path} exist.")
        # Create dummy files for initialization testing
        os.makedirs('data', exist_ok=True)
        pd.DataFrame({
            'case_id': ['patient1', 'patient2'],
            'slide_path': ['file:///tmp/dummy1.svs', 's3://dummy-bucket/dummy2.svs'],
            'survival_time': [100.5, 45.2],
            'censor': [1, 0]
        }).to_csv(csv_path, index=False)
        
        pd.DataFrame({
            'patient1': [10, 20, 30],
            'patient2': [5, 10, 15]
        }, index=['GENE1', 'GENE2', 'GENE3']).to_csv(tsv_path, sep='\t')
        print("Created dummy datasets for testing.")

    # 2. Data Loading
    print("Loading datasets...")
    dataset = DummyMultiModalDataset(csv_path, tsv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    num_genes = len(dataset.genomic_loader.data)
    gene_list = dataset.genomic_loader.data.index.tolist()
    
    mapper = PathwayMapper(gene_list)
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
        device=device
    )
    
    # 6. Training Loop
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch(dataloader, alpha=0.5)
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

if __name__ == '__main__':
    main()
