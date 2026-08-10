import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PathwayAwareTransformer(nn.Module):
    def __init__(self, num_genes, num_pathways, pathway_mask, d_model=128, nhead=4, num_layers=2):
        super(PathwayAwareTransformer, self).__init__()
        self.num_genes = num_genes
        self.num_pathways = num_pathways
        
        # Buffer to keep the binary mask (num_pathways, num_genes)
        self.register_buffer('pathway_mask', pathway_mask)
        
        # Sparse linear layer parameters mapping genes to pathways
        self.gene_to_pathway_weight = nn.Parameter(torch.Tensor(num_pathways, num_genes))
        self.gene_to_pathway_bias = nn.Parameter(torch.Tensor(num_pathways))
        self._init_sparse_weights()
        
        # Project scalar pathway activations to embedding dimension
        self.pathway_proj = nn.Linear(1, d_model)
        
        # Transformer to model pathway-pathway interactions
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection to genomic embedding
        self.fc_out = nn.Linear(d_model, d_model)
        
    def _init_sparse_weights(self):
        nn.init.kaiming_uniform_(self.gene_to_pathway_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.gene_to_pathway_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.gene_to_pathway_bias, -bound, bound)

    def forward(self, x):
        # x: (batch_size, num_genes)
        
        # Enforce sparsity by masking the weights with pathway_mask
        masked_weight = self.gene_to_pathway_weight * self.pathway_mask
        
        # pathway_act: (batch_size, num_pathways)
        pathway_act = F.linear(x, masked_weight, self.gene_to_pathway_bias)
        pathway_act = F.relu(pathway_act)
        
        # Project each pathway scalar to d_model tokens
        # tokens: (batch_size, num_pathways, d_model)
        tokens = self.pathway_proj(pathway_act.unsqueeze(-1))
        
        # Apply transformer
        out = self.transformer(tokens)
        
        # Aggregate pathways (mean pooling)
        out = out.mean(dim=1) # (batch_size, d_model)
        
        # Produce final genomic embeddings
        emb = self.fc_out(out)
        
        return emb
