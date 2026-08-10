import unittest
import os
import tempfile
import numpy as np
import torch
import sys
from PIL import Image
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.pathology import WSIPipeline, PatchFeatureExtractor
from models.pathology_branch import WSI_MIL_Encoder

class TestPathologyBranch(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.patch_dir = os.path.join(self.temp_dir.name, 'mock_patches')
        os.makedirs(self.patch_dir, exist_ok=True)
        
        # Create some dummy 256x256 patches
        for i in range(10): # Just 10 patches for testing
            dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(dummy_img)
            img.save(os.path.join(self.patch_dir, f'patch_{i}.png'))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_wsi_pipeline_mocked(self):
        # We mock S3 download and openslide logic
        pipeline = WSIPipeline(output_dir=self.temp_dir.name)
        
        # Mock download
        def mock_download(s3_url, local_path):
            with open(local_path, 'w') as f:
                f.write("mocksvs")
            return local_path
        pipeline.download_from_s3 = mock_download
        
        # Mock extract
        def mock_extract(slide_path, patient_id):
            return 5 # returns patch count
        pipeline.extract_patches = mock_extract
        
        pipeline.process_wsi("s3://fake-bucket/fake.svs", "patient_01")
        # Ensure raw WSI was deleted (temp file)
        self.assertFalse(os.path.exists('/tmp/fake.svs'))

    def test_dino_feature_extraction(self):
        # We test the feature extractor and k-means clustering
        # Set num_clusters to 5 for speed with 10 dummy patches
        extractor = PatchFeatureExtractor(model_name='dino_vits16', num_clusters=5, 
                                          output_dir=self.temp_dir.name, device='cpu')
        
        out_path = extractor.process_patient("patient_02", self.patch_dir, delete_patches=False)
        self.assertIsNotNone(out_path)
        
        features = torch.load(out_path)
        
        # Should be 5 clusters, dino_vits16 has dimension 384
        self.assertEqual(features.shape, (5, 384))
        self.assertFalse(torch.isnan(features).any())

    def test_vision_transformer_encoder(self):
        batch_size = 4
        num_patches = 500
        input_dim = 384
        d_model = 128
        
        model = WSI_MIL_Encoder(input_dim=input_dim, d_model=d_model, num_patches=num_patches)
        
        # Dummy input
        x = torch.randn(batch_size, num_patches, input_dim)
        
        out = model(x)
        
        self.assertEqual(out.shape, (batch_size, d_model))
        self.assertFalse(torch.isnan(out).any())

if __name__ == '__main__':
    unittest.main()
