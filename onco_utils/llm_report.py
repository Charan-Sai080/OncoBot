import requests
import json

class ClinicalReportGenerator:
    def __init__(self, model="minimax-m3"):
        self.model = model
        self.ollama_url = "https://api.ollama.com/api/generate"
        self.api_key = "fe01f3e52c9045bf91df5d9cdf14cb06.tBiJRdgUUmfmNk96fWd7PP8K"

    def generate_prompt(self, patient_id, top_pathways, attention_summary, survival_score):
        """
        Formats the model findings into a clear text prompt for the LLM.
        """
        prompt = f"""
        You are an expert AI Pathology and Genomics Assistant. 
        Please generate a comprehensive but concise clinical report summarizing the multimodal findings for the following patient.

        Patient ID: {patient_id}
        Predicted Survival Risk Score (Cox-PH): {survival_score:.4f}

        Top Activated Biological Pathways (Genomics):
        {', '.join(top_pathways)}

        Pathology Cross-Attention Highlights:
        {attention_summary}

        Please include:
        1. An executive summary.
        2. A brief analysis of the interplay between the highlighted pathways and the morphological features attended to in the WSI.
        3. Potential clinical implications of the survival risk score.
        """
        return prompt.strip()

    def generate_report(self, patient_id, top_pathways, attention_summary, survival_score):
        """
        Calls the cloud Ollama.com API to generate the human-readable clinical report.
        """
        prompt = self.generate_prompt(patient_id, top_pathways, attention_summary, survival_score)
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": "You are a clinical AI reporting assistant.",
                "stream": False
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.post(self.ollama_url, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                return response.json().get("response", "Error: No response generated.")
            else:
                return f"Error from Ollama: {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to local Ollama (is it running?): {str(e)}"
