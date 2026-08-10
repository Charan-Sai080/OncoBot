import os
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class ClinicalReportGenerator:
    def __init__(self, api_key=None, model="gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        if OpenAI is not None and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

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
        Calls the LLM API to generate the human-readable clinical report.
        """
        prompt = self.generate_prompt(patient_id, top_pathways, attention_summary, survival_score)
        
        if self.client is None:
            # Fallback if API key or library is missing
            return f"[MOCK REPORT]\nPrompt sent to LLM:\n{prompt}\n\n(Install openai and set OPENAI_API_KEY to generate real reports.)"
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical AI reporting assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating report: {str(e)}"
