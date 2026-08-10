import pandas as pd
from torch.utils.data import Dataset
from .genomic import GenomicDataLoader

class MultiModalDataset(Dataset):
    def __init__(self, csv_path, tsv_path):
        self.slide_data = pd.read_csv(csv_path)
        self.genomic_loader = GenomicDataLoader(tsv_path)
        
        self.aligned_data = self._align_data()

    def _align_data(self):
        aligned = []
        patient_col = 'case_id' if 'case_id' in self.slide_data.columns else self.slide_data.columns[0]
        
        for _, row in self.slide_data.iterrows():
            patient_id = row[patient_col]
            # often TCGA IDs in CSV might need formatting to match TSV
            genomic_features = self.genomic_loader.get_patient_data(patient_id)
            if genomic_features is not None:
                aligned.append({
                    'patient_id': patient_id,
                    'slide_info': row.to_dict(),
                    'genomic_features': genomic_features.values
                })
        return aligned

    def __len__(self):
        return len(self.aligned_data)

    def __getitem__(self, idx):
        return self.aligned_data[idx]
