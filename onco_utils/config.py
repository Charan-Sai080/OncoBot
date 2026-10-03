import os
from dataclasses import dataclass

def _load_env_file():
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass

_load_env_file()

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
    
    # Hugging Face LLM configuration for Clinical Decision Support Reporting
    hf_token: str = os.getenv("HF_TOKEN", "")
    hf_model: str = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    hf_router_url: str = "https://router.huggingface.co/v1/chat/completions"


    # Backwards-compatibility aliases
    ollama_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    ollama_url: str = "https://router.huggingface.co/v1/chat/completions"

config = OncoBotConfig()

