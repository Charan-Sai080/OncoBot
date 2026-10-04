from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import boto3
import uuid
import json
import torch
import numpy as np

# In a real app, these would be imported from your project root
# import sys
# sys.path.append('..')
# from models.fusion import CrossAttentionFusion

app = FastAPI(title="OncoBot Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mocked S3 client for generating pre-signed URLs
# s3_client = boto3.client('s3', region_name='us-east-1')
BUCKET_NAME = "oncobot-wsi-uploads"

@app.get("/upload-url")
def get_presigned_url(filename: str):
    """
    Step 1: The browser asks for permission to upload directly to AWS S3.
    This bypasses our Python backend entirely so we don't crash from 4GB files.
    """
    object_name = f"uploads/{uuid.uuid4()}_{filename}"
    
    # In a real scenario, you'd generate the AWS URL:
    # url = s3_client.generate_presigned_url('put_object', 
    #                                        Params={'Bucket': BUCKET_NAME, 'Key': object_name}, 
    #                                        ExpiresIn=3600)
    
    # Mocking for this prototype
    url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{object_name}?AWSAccessKeyId=MOCK&Signature=MOCK"
    
    return {
        "upload_url": url,
        "s3_key": object_name
    }

def generate_heatmap_task(patient_id: str, s3_key: str):
    """
    Background Task (Step 2): 
    This runs after the upload finishes. It converts the WSI to DeepZoom and calculates Heatmap attention.
    """
    output_dir = f"static/patients/{patient_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[{patient_id}] 1. Downloading WSI from S3 ({s3_key}) to local disk...")
    # s3_client.download_file(BUCKET_NAME, s3_key, local_wsi_path)
    
    print(f"[{patient_id}] 2. Generating Deep Zoom Image (DZI) tiles using libvips/openslide...")
    # e.g., subprocess.run(["vips", "dzsave", local_wsi_path, f"{output_dir}/slide"])
    
    print(f"[{patient_id}] 3. Running DINO & Cross-Attention Model to get Attention Weights...")
    # fused_emb, attn_weights = fusion_model(g_emb, p_emb, return_attention=True)
    # attn_weights shape: [batch, query_len (pathways), key_len (WSI patches)]
    
    # MOCKING the coordinates and attention scores that would come from WSIPipeline
    # Assuming 1024 patches were extracted, map their (x,y) to their attention score
    heatmap_data = []
    
    for i in range(1024):
        # Mock grid positions
        x = (i % 32) * 256
        y = (i // 32) * 256
        
        # Mock attention score between 0.0 and 1.0 (emphasizing the middle)
        score = np.random.beta(0.5, 0.5) 
        
        heatmap_data.append({
            "id": i,
            "x": x,
            "y": y,
            "width": 256,
            "height": 256,
            "attention_score": float(score)
        })
        
    heatmap_path = os.path.join(output_dir, "heatmap.json")
    with open(heatmap_path, "w") as f:
        json.dump({"patient_id": patient_id, "patches": heatmap_data}, f)
        
    print(f"[{patient_id}] Heatmap JSON generated and saved to {heatmap_path}")

@app.post("/process-wsi")
def process_wsi(patient_id: str, s3_key: str, background_tasks: BackgroundTasks):
    """
    Triggers the background generation of the DeepZoom tiles and the JSON heatmap overlay.
    """
    background_tasks.add_task(generate_heatmap_task, patient_id, s3_key)
    return {"message": f"Processing started for {patient_id}. Tiles and Heatmap will be available soon."}

@app.get("/heatmap-data/{patient_id}")
def get_heatmap(patient_id: str):
    """
    Step 3: The browser fetches this lightweight JSON to paint the red/blue squares over the DeepZoom tiles.
    """
    heatmap_path = f"static/patients/{patient_id}/heatmap.json"
    if not os.path.exists(heatmap_path):
        raise HTTPException(status_code=404, detail="Heatmap not generated yet.")
        
    with open(heatmap_path, "r") as f:
        data = json.load(f)
        
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
