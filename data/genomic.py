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
    def __init__(self, gene_list, pathway_db='kegg'):
        self.gene_list = gene_list
        self.num_genes = len(gene_list)
        # Using 50 placeholder pathways for initialization
        self.num_pathways = 50 
        self.pathway_mask = self._build_dummy_mask()
        
    def _build_dummy_mask(self):
        import torch
        # Binary mask: (num_pathways, num_genes)
        # 1 if gene is in pathway, else 0
        mask = np.zeros((self.num_pathways, self.num_genes))
        for j in range(self.num_genes):
            p_idx = np.random.choice(self.num_pathways, size=np.random.randint(1, 4), replace=False)
            mask[p_idx, j] = 1
        return torch.tensor(mask, dtype=torch.float32)
