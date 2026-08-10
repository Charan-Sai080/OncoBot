import torch
import torch.nn as nn

class WSI_MIL_Encoder(nn.Module):
    def __init__(self, input_dim=384, d_model=128, nhead=8, num_layers=4, num_patches=1024):
        super(WSI_MIL_Encoder, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Projection from DINO dimension to model dimension
        self.proj = nn.Linear(input_dim, d_model)
        
        # Learnable class token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Position embeddings (optional, but good for MIL if positional information is retained, 
        # though WSI patches are generally permutation invariant unless coordinates are explicitly encoded. 
        # We add it just in case, or leave it out for permutation invariance.)
        # We'll omit positional embeddings to treat it as a true bag of features (MIL).
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection layer
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: (batch_size, num_patches, input_dim)
        batch_size = x.size(0)
        
        # 1. Project patch embeddings
        x = self.proj(x) # (batch_size, num_patches, d_model)
        
        # 2. Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch_size, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1) # (batch_size, num_patches + 1, d_model)
        
        # 3. Apply Transformer
        out = self.transformer(x) # (batch_size, num_patches + 1, d_model)
        
        # 4. Extract CLS token output for patient-level embedding
        cls_out = out[:, 0, :] # (batch_size, d_model)
        
        # 5. Final FC layer
        emb = self.fc_out(cls_out) # (batch_size, d_model)
        
        return emb
