import torch
import torch.nn as nn

class CoxSurvivalHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=[64, 32], dropout=0.2):
        super(CoxSurvivalHead, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Build hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim
            
        # Final output layer to predict a single risk score (scalar) per patient
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, fused_features):
        """
        fused_features: Representation from the multimodal fusion module (batch_size, input_dim)
        Returns:
            risk_scores: Predicted risk scores (batch_size, 1)
        """
        risk_scores = self.network(fused_features)
        return risk_scores
