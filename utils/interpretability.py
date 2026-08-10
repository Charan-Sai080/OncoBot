import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class CrossAttentionInterpreter:
    def __init__(self, fusion_model, device='cpu'):
        self.fusion_model = fusion_model.to(device)
        self.device = device

    def extract_attention_weights(self, genomic_emb, pathology_emb):
        """
        Extract cross-attention weights between Genomics (Query) and Pathology (Key/Value).
        Returns the attention weights of shape (batch_size, num_heads, seq_len_g, seq_len_p).
        """
        self.fusion_model.eval()
        with torch.no_grad():
            if genomic_emb.dim() == 2:
                genomic_emb = genomic_emb.unsqueeze(1)
            if pathology_emb.dim() == 2:
                pathology_emb = pathology_emb.unsqueeze(1)
                
            query = self.fusion_model.norm_g(genomic_emb.to(self.device))
            key = self.fusion_model.norm_p(pathology_emb.to(self.device))
            value = key
            
            # Need to get attention weights from MultiheadAttention
            # Unfortunately, by default nn.MultiheadAttention returns average attention weights across heads
            # if we pass need_weights=True. But let's return it directly.
            attn_output, attn_weights = self.fusion_model.multihead_attn(
                query, key, value, need_weights=True, average_attn_weights=False
            )
            
        return attn_weights.cpu().numpy()

    def generate_heatmap(self, attention_weights, patch_paths, output_dir, patient_id):
        """
        attention_weights: (batch_size, num_heads, 1, num_patches)
        patch_paths: list of paths to original .png patches
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Average across heads and batch, and sequence length of genomic queries (assumed 1)
        # Resulting in 1D array of weights for each patch
        avg_weights = np.mean(attention_weights, axis=(0, 1, 2))
        
        # Normalize weights for visualization
        avg_weights = (avg_weights - avg_weights.min()) / (avg_weights.max() - avg_weights.min() + 1e-8)
        
        # Overlay heatmap on original patches
        for idx, (weight, patch_path) in enumerate(zip(avg_weights, patch_paths)):
            if not os.path.exists(patch_path):
                continue
                
            img = cv2.imread(patch_path)
            if img is None:
                continue
                
            # Create heatmap colored overlay based on the weight
            heatmap = cv2.applyColorMap(np.uint8(255 * weight * np.ones_like(img[:, :, 0])), cv2.COLORMAP_JET)
            
            # Blend
            alpha = 0.5
            blended = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
            
            out_name = f"attn_{patient_id}_patch_{idx}_w{weight:.2f}.png"
            cv2.imwrite(os.path.join(output_dir, out_name), blended)
            
        return avg_weights
