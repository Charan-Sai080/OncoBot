import os
from dataclasses import dataclass

@dataclass
class OncoBotConfig:
    # Server configuration
    host: str = "127.0.0.1"
    port: int = 8000
    
    # Model Hyperparameters
    d_model: int = 128
    num_heads: int = 4
    num_patches: int = 1024
    dino_dim: int = 384
    
    # Paths
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gmt_path: str = os.path.join(base_dir, "Datasets", "kegg_cancer_pathways.gmt")
    wsi_features_dir: str = os.path.join(base_dir, "Datasets", "WSI_Features")
    cache_dir: str = os.path.join(base_dir, "cache")
    static_dir: str = os.path.join(base_dir, "ui")
    
    # Default patient demo
    default_patient_id: str = "TCGA-2F-A9KO"
    cancer_type: str = "TCGA-BLCA (Bladder Urothelial Carcinoma)"
    
    # LLM configuration
    ollama_model: str = "minimax-m3"
    ollama_url: str = "https://api.ollama.com/api/generate"

config = OncoBotConfig()
