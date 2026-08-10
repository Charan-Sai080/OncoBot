import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features_g, features_p):
        """
        features_g: Genomic embeddings (batch_size, dim)
        features_p: Pathology embeddings (batch_size, dim)
        """
        # Normalize features
        features_g = F.normalize(features_g, dim=1)
        features_p = F.normalize(features_p, dim=1)
        
        batch_size = features_g.shape[0]
        
        # Calculate logits
        # (batch_size, batch_size)
        logits_g2p = torch.matmul(features_g, features_p.T) / self.temperature
        logits_p2g = torch.matmul(features_p, features_g.T) / self.temperature
        
        labels = torch.arange(batch_size, device=features_g.device)
        
        # Cross entropy loss
        loss_g2p = F.cross_entropy(logits_g2p, labels)
        loss_p2g = F.cross_entropy(logits_p2g, labels)
        
        loss = (loss_g2p + loss_p2g) / 2
        return loss

class CoxPHLoss(nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()
        
    def forward(self, risk_scores, survivals, censors):
        """
        risk_scores: Predicted risk scores from the model (batch_size, 1)
        survivals: True survival times (batch_size, )
        censors: Event indicator (1 if event occurred, 0 if censored) (batch_size, )
        """
        # Sort data by descending survival time
        survivals = survivals.squeeze()
        censors = censors.squeeze()
        risk_scores = risk_scores.squeeze()
        
        sorted_indices = torch.argsort(survivals, descending=True)
        survivals = survivals[sorted_indices]
        censors = censors[sorted_indices]
        risk_scores = risk_scores[sorted_indices]
        
        # Calculate risk set (sum of exp(risk_score) for all j >= i)
        # Using log-sum-exp trick for numerical stability
        exp_risk = torch.exp(risk_scores)
        risk_set_sum = torch.cumsum(exp_risk, dim=0)
        
        # Negative log partial likelihood
        # \sum_{i: censor_i=1} (risk_score_i - log(risk_set_sum_i))
        log_risk_set = torch.log(risk_set_sum)
        
        # Only compute loss for uncensored events
        uncensored_loss = risk_scores - log_risk_set
        loss = -torch.sum(uncensored_loss * censors)
        
        # Average over number of events
        num_events = torch.sum(censors)
        if num_events > 0:
            loss = loss / num_events
            
        return loss
