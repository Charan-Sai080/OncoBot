import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.d_model = d_model
        
        # Multihead cross attention
        # Note: PyTorch MultiheadAttention expects (L, N, E) or (N, L, E) if batch_first=True
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, 
                                                    dropout=dropout, batch_first=True)
        
        # Optional projection layers if dimensions need to match d_model
        # We assume inputs are already projected to d_model, but let's add layer norms
        self.norm_g = nn.LayerNorm(d_model)
        self.norm_p = nn.LayerNorm(d_model)
        
        # Final feed-forward network to aggregate the fused representation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, genomic_emb, pathology_emb):
        """
        genomic_emb: (batch_size, seq_len_g, d_model) or (batch_size, d_model)
        pathology_emb: (batch_size, seq_len_p, d_model) or (batch_size, d_model)
        """
        # Ensure sequence dimension exists
        if genomic_emb.dim() == 2:
            genomic_emb = genomic_emb.unsqueeze(1) # (batch_size, 1, d_model)
        if pathology_emb.dim() == 2:
            pathology_emb = pathology_emb.unsqueeze(1) # (batch_size, 1, d_model)
            
        # Apply layer norm
        query = self.norm_g(genomic_emb)
        key = self.norm_p(pathology_emb)
        value = key
        
        # Cross Attention: Genomics is Query, Pathology is Key/Value
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        
        # Residual connection and FFN
        fused = query + attn_output
        fused = self.ffn(fused)
        
        # If the output sequence length is 1, squeeze it back
        # We generally want a single feature vector per patient
        if fused.size(1) == 1:
            fused = fused.squeeze(1)
            
        # If sequence length > 1, we could apply pooling (e.g. mean pooling)
        elif fused.dim() == 3:
            fused = fused.mean(dim=1)
            
        return fused
