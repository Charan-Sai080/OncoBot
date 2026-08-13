import pandas as pd
import numpy as np

class GenomicDataLoader:
    def __init__(self, tsv_path):
        self.tsv_path = tsv_path
        self.data = self._load_and_normalize()

    def _load_and_normalize(self):
        df = pd.read_csv(self.tsv_path, sep='\t', index_col=0)
        
        # Log-normalization (log2(x + 1))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = np.log2(df[numeric_cols] + 1)
        
        return df

    def get_patient_data(self, patient_id):
        # Basic check for exact match
        if patient_id in self.data.columns:
            return self.data[patient_id]
        elif patient_id in self.data.index:
            return self.data.loc[patient_id]
            
        # TCGA ids sometimes have variations (e.g. TCGA-XX-XXXX-01A vs TCGA-XX-XXXX)
        for col in self.data.columns:
            if patient_id in str(col):
                return self.data[col]
                
        for idx in self.data.index:
            if patient_id in str(idx):
                return self.data.loc[idx]
                
        return None

class PathwayMapper:
    def __init__(self, gene_list, gmt_path=None):
        self.gene_list = gene_list
        self.num_genes = len(gene_list)
        
        # We need a fast lookup for gene index
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}
        
        self.pathway_names = []
        
        if gmt_path:
            self.pathway_mask = self._build_mask_from_gmt(gmt_path)
        else:
            self.num_pathways = 50
            self.pathway_mask = self._build_dummy_mask()
            
    def _build_mask_from_gmt(self, gmt_path):
        import torch
        import os
        
        if not os.path.exists(gmt_path):
            raise FileNotFoundError(f"GMT file not found at {gmt_path}")
            
        # Parse the GMT file
        pathways = []
        with open(gmt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    pathway_name = parts[0]
                    # parts[1] is URL/description
                    genes_in_pathway = parts[2:]
                    pathways.append((pathway_name, genes_in_pathway))
                    self.pathway_names.append(pathway_name)
                    
        self.num_pathways = len(pathways)
        mask = np.zeros((self.num_pathways, self.num_genes))
        
        # Build the boolean matrix mapping genes to pathways
        for p_idx, (p_name, p_genes) in enumerate(pathways):
            for gene in p_genes:
                if gene in self.gene_to_idx:
                    g_idx = self.gene_to_idx[gene]
                    mask[p_idx, g_idx] = 1
                    
        return torch.tensor(mask, dtype=torch.float32)

    def _build_dummy_mask(self):
        import torch
        # Binary mask: (num_pathways, num_genes)
        mask = np.zeros((self.num_pathways, self.num_genes))
        for j in range(self.num_genes):
            p_idx = np.random.choice(self.num_pathways, size=np.random.randint(1, 4), replace=False)
            mask[p_idx, j] = 1
        
        self.pathway_names = [f"DUMMY_PATHWAY_{i}" for i in range(self.num_pathways)]
        return torch.tensor(mask, dtype=torch.float32)
