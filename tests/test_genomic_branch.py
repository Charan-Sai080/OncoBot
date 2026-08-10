import unittest
import os
import tempfile
import pandas as pd
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.genomic import GenomicDataLoader, PathwayMapper
from models.genomic_branch import PathwayAwareTransformer

class TestGenomicBranch(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tsv_path = os.path.join(self.temp_dir.name, 'mock_counts.tsv')
        
        self.mock_data = pd.DataFrame({
            'TCGA-01': [10.0, 0.0, 50.0],
            'TCGA-02': [0.0, 20.0, 5.0],
            'TCGA-03': [100.0, 10.0, 0.0]
        }, index=['ENSG001', 'ENSG002', 'ENSG003'])
        
        self.mock_data.to_csv(self.tsv_path, sep='\t')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_genomic_data_loader(self):
        loader = GenomicDataLoader(self.tsv_path)
        self.assertEqual(loader.data.shape, (3, 3))
        
        expected_val = np.log2(10.0 + 1)
        self.assertAlmostEqual(loader.data.loc['ENSG001', 'TCGA-01'], expected_val, places=4)
        
        patient_data = loader.get_patient_data('TCGA-02')
        self.assertIsNotNone(patient_data)
        self.assertEqual(len(patient_data), 3)

    def test_pathway_mapper(self):
        genes = ['ENSG001', 'ENSG002', 'ENSG003', 'ENSG004', 'ENSG005']
        mapper = PathwayMapper(genes)
        
        self.assertEqual(mapper.num_genes, 5)
        self.assertEqual(mapper.num_pathways, 50)
        self.assertEqual(mapper.pathway_mask.shape, (50, 5))
        
        unique_vals = torch.unique(mapper.pathway_mask)
        for val in unique_vals:
            self.assertIn(val.item(), [0.0, 1.0])

    def test_pathway_aware_transformer(self):
        num_genes = 100
        num_pathways = 50
        batch_size = 16
        d_model = 64
        
        mask = torch.randint(0, 2, (num_pathways, num_genes)).float()
        
        model = PathwayAwareTransformer(
            num_genes=num_genes, 
            num_pathways=num_pathways, 
            pathway_mask=mask, 
            d_model=d_model
        )
        
        x = torch.randn(batch_size, num_genes)
        output = model(x)
        
        self.assertEqual(output.shape, (batch_size, d_model))
        self.assertFalse(torch.isnan(output).any())

if __name__ == '__main__':
    unittest.main()
