import os
import boto3
import cv2
import numpy as np
from urllib.parse import urlparse
try:
    import openslide
except ImportError:
    openslide = None

class WSIPipeline:
    """
    Pipeline for acquiring WSI from S3, performing tissue segmentation,
    extracting patches, and deleting the raw WSI to manage hardware limitations.
    """
    def __init__(self, patch_size=256, output_dir='patches', threshold=210):
        self.patch_size = patch_size
        self.output_dir = output_dir
        self.threshold = threshold # Intensity threshold for background
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        # Initialize AWS S3 client
        self.s3 = boto3.client('s3')

    def download_from_s3(self, s3_url, local_path):
        """Downloads an S3 object to a local path."""
        parsed_url = urlparse(s3_url)
        bucket = parsed_url.netloc
        key = parsed_url.path.lstrip('/')
        
        print(f"Downloading s3://{bucket}/{key} to {local_path}...")
        self.s3.download_file(bucket, key, local_path)
        return local_path

    def segment_tissue(self, slide):
        """
        Segment the tissue from the slide by creating a coarse tissue mask.
        Returns the mask at a lower magnification.
        """
        # Read slide at lowest magnification for quick segmentation
        level = slide.level_count - 1
        dimensions = slide.level_dimensions[level]
        
        rgba = slide.read_region((0, 0), level, dimensions)
        rgb = np.array(rgba)[:, :, :3]
        
        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Simple thresholding to ignore white background
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        
        return mask, level

    def extract_patches(self, slide_path, patient_id):
        """
        Extract patches from a slide and save them to disk.
        """
        if openslide is None:
            raise ImportError("openslide-python is required for WSI processing.")
            
        slide = openslide.OpenSlide(slide_path)
        
        mask, mask_level = self.segment_tissue(slide)
        mask_downsample = slide.level_downsamples[mask_level]
        
        width, height = slide.level_dimensions[0]
        
        patient_out_dir = os.path.join(self.output_dir, patient_id)
        os.makedirs(patient_out_dir, exist_ok=True)
        
        patch_count = 0
        
        # Iterate over the slide in grid fashion
        for y in range(0, height, self.patch_size):
            for x in range(0, width, self.patch_size):
                # Calculate corresponding coordinates in the downsampled mask
                mask_x = int(x / mask_downsample)
                mask_y = int(y / mask_downsample)
                mask_w = max(1, int(self.patch_size / mask_downsample))
                mask_h = max(1, int(self.patch_size / mask_downsample))
                
                # Ensure we don't go out of bounds on the mask
                if mask_y + mask_h > mask.shape[0] or mask_x + mask_w > mask.shape[1]:
                    continue
                
                # Check if the region has enough tissue
                region_mask = mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w]
                if np.mean(region_mask) > 10: # If tissue covers a sufficient portion
                    patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
                    patch_rgb = np.array(patch)[:, :, :3]
                    
                    patch_filename = os.path.join(patient_out_dir, f"{patient_id}_x{x}_y{y}.png")
                    cv2.imwrite(patch_filename, cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
                    patch_count += 1
        
        slide.close()
        return patch_count

    def process_wsi(self, s3_url, patient_id):
        """
        Main pipeline method:
        1. Download WSI
        2. Extract Patches
        3. Delete WSI to save space
        """
        filename = os.path.basename(s3_url)
        cache_dir = os.environ.get('ONCO_BOT_CACHE_DIR', '/tmp')
        temp_slide_path = os.path.join(cache_dir, filename)
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        try:
            self.download_from_s3(s3_url, temp_slide_path)
            print(f"Extracting patches for {patient_id}...")
            num_patches = self.extract_patches(temp_slide_path, patient_id)
            print(f"Extracted {num_patches} patches.")
        finally:
            # Clean up the raw WSI to respect hardware limitations
            if os.path.exists(temp_slide_path):
                os.remove(temp_slide_path)
                print(f"Deleted raw WSI file {temp_slide_path} to conserve space.")

import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
import glob
import shutil

class PatchFeatureExtractor:
    def __init__(self, model_name='dino_vits16', num_clusters=500, output_dir='features', device=None):
        self.num_clusters = num_clusters
        self.output_dir = output_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained DINO model
        self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def extract_features(self, patch_dir):
        """Extract DINO features for all patches in a directory."""
        patch_files = glob.glob(os.path.join(patch_dir, '*.png'))
        features = []
        
        with torch.no_grad():
            for pf in patch_files:
                img = Image.open(pf).convert('RGB')
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                feat = self.model(tensor)
                features.append(feat.cpu().squeeze(0).numpy())
                
        return np.array(features), patch_files

    def process_patient(self, patient_id, patch_dir, delete_patches=True):
        """
        Extract features, run K-Means to sample exactly 500 patches, 
        and save to disk as a tensor.
        """
        features, patch_files = self.extract_features(patch_dir)
        
        if len(features) == 0:
            print(f"No patches found for {patient_id}.")
            return None
            
        # Pad with zeros if less than num_clusters patches
        if len(features) < self.num_clusters:
            pad_len = self.num_clusters - len(features)
            pad_feats = np.zeros((pad_len, features.shape[1]), dtype=np.float32)
            sampled_features = np.vstack((features, pad_feats))
        else:
            # K-Means clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init='auto')
            kmeans.fit(features)
            
            # Find the closest patch to each cluster center
            sampled_features = []
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(features - center, axis=1)
                closest_idx = np.argmin(distances)
                sampled_features.append(features[closest_idx])
            sampled_features = np.array(sampled_features)
            
        # Save as PyTorch tensor
        out_tensor = torch.tensor(sampled_features, dtype=torch.float32)
        out_path = os.path.join(self.output_dir, f"{patient_id}_features.pt")
        torch.save(out_tensor, out_path)
        print(f"Saved 500 patch features to {out_path}.")
        
        # Optionally delete raw patches to save disk space
        if delete_patches and os.path.exists(patch_dir):
            shutil.rmtree(patch_dir)
            print(f"Deleted raw patches for {patient_id} to conserve disk space.")
            
        return out_path
