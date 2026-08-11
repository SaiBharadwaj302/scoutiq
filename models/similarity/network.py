"""
models/similarity/network.py

Feed-forward embedding network for player similarity. Outputs an
L2-normalized vector so retrieval can use plain dot-product / cosine
similarity in the API layer.
"""
import torch.nn as nn
import torch.nn.functional as F


class PlayerEmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        # L2 normalize for cosine similarity mapping
        return F.normalize(x, p=2, dim=1)
