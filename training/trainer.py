import os
import torch
from tqdm import tqdm
from data.pathology import WSIPipeline, PatchFeatureExtractor

class MultimodalTrainer:
    def __init__(
        self, 
        genomic_model, 
        pathology_model, 
        fusion_model, 
        survival_head,
        info_nce_loss_fn,
        cox_loss_fn,
        optimizer,
        device='cpu',
        wsi_output_dir='./wsi_patches',
        features_output_dir='./wsi_features'
    ):
        self.genomic_model = genomic_model.to(device)
        self.pathology_model = pathology_model.to(device)
        self.fusion_model = fusion_model.to(device)
        self.survival_head = survival_head.to(device)
        
        self.info_nce_loss_fn = info_nce_loss_fn
        self.cox_loss_fn = cox_loss_fn
        self.optimizer = optimizer
        self.device = device
        
        self.wsi_pipeline = WSIPipeline(output_dir=wsi_output_dir)
        self.feature_extractor = PatchFeatureExtractor(output_dir=features_output_dir, device=device)

    def _get_or_extract_wsi_features(self, patient_id, wsi_path):
        """Check or extract features for a single patient."""
        feat_path = os.path.join(self.feature_extractor.output_dir, f"{patient_id}_features.pt")
        
        if os.path.exists(feat_path):
            return torch.load(feat_path, map_location=self.device)
            
        print(f"Extracting features for {patient_id} from {wsi_path}...")
        
        if wsi_path.startswith('s3://'):
            cache_dir = os.environ.get('ONCO_BOT_CACHE_DIR', '/tmp')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            local_slide_path = os.path.join(cache_dir, os.path.basename(wsi_path))
            try:
                self.wsi_pipeline.download_from_s3(wsi_path, local_slide_path)
                self.wsi_pipeline.extract_patches(local_slide_path, patient_id)
            finally:
                if os.path.exists(local_slide_path):
                    os.remove(local_slide_path)
        else:
            local_slide_path = wsi_path.replace('file://', '')
            self.wsi_pipeline.extract_patches(local_slide_path, patient_id)
            
        patient_patch_dir = os.path.join(self.wsi_pipeline.output_dir, patient_id)
        saved_feat_path = self.feature_extractor.process_patient(patient_id, patient_patch_dir, delete_patches=True)
        
        if saved_feat_path is None:
            return torch.zeros((500, 384)).to(self.device)
            
        return torch.load(saved_feat_path, map_location=self.device)

    def train_epoch(self, dataloader, alpha=0.5):
        self.genomic_model.train()
        self.pathology_model.train()
        self.fusion_model.train()
        self.survival_head.train()
        
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            patient_ids = batch['patient_id']
            genomic_features = batch['genomic_features'].float().to(self.device)
            wsi_paths = batch['wsi_path']
            survivals = batch['survival_time'].float().to(self.device)
            censors = batch['censor'].float().to(self.device)
            
            # Load pathology features dynamically to prevent OOM and allow batching
            pathology_patches_list = []
            for pid, path in zip(patient_ids, wsi_paths):
                p_feats = self._get_or_extract_wsi_features(pid, path)
                pathology_patches_list.append(p_feats)
                
            pathology_patches = torch.stack(pathology_patches_list).to(self.device) # (B, 500, input_dim)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            g_emb = self.genomic_model(genomic_features) # (B, d_model)
            p_emb = self.pathology_model(pathology_patches) # (B, d_model)
            
            # Loss 1: Alignment
            loss_align = self.info_nce_loss_fn(g_emb, p_emb)
            
            # Fusion
            fused = self.fusion_model(g_emb, p_emb) # (B, d_model)
            
            # Survival Prediction
            risk_scores = self.survival_head(fused) # (B, 1)
            
            # Loss 2: Cox PH
            loss_cox = self.cox_loss_fn(risk_scores, survivals, censors)
            
            # Total Loss
            loss = (alpha * loss_align) + ((1 - alpha) * loss_cox)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(dataloader)
