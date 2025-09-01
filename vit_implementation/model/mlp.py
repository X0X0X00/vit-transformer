import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MLPPlusPlus(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        # First MLP branch
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
        # Second MLP branch with different activation
        self.fc3 = nn.Linear(embed_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embed_dim)
        
        # Activations
        self.gelu = nn.GELU()
        self.swish = nn.SiLU()  # Swish activation
        
        self.dropout = nn.Dropout(dropout)
        
        # Gate mechanism
        self.gate = nn.Linear(embed_dim, 2)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # First branch with GELU
        branch1 = self.fc1(x)
        branch1 = self.gelu(branch1)
        branch1 = self.dropout(branch1)
        branch1 = self.fc2(branch1)
        branch1 = self.dropout(branch1)
        
        # Second branch with Swish
        branch2 = self.fc3(x)
        branch2 = self.swish(branch2)
        branch2 = self.dropout(branch2)
        branch2 = self.fc4(branch2)
        branch2 = self.dropout(branch2)
        
        # Gating mechanism
        gate_weights = self.gate(x)
        gate_weights = self.softmax(gate_weights)
        
        # Weighted combination
        output = gate_weights[..., 0:1] * branch1 + gate_weights[..., 1:2] * branch2
        
        return output