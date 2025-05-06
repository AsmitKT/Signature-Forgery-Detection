# signature_embedding_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SignatureEmbeddingCNN(nn.Module):
    """
    CNN that maps a 1×H×W signature image → a D‑dim embedding vector.
    Revised to use Global Average Pooling, BatchNorm, LeakyReLU, stronger dropout,
    and proper weight initialization to avoid embedding collapse.
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)

        # Global average pooling to reduce spatial dims
        self.gap    = nn.AdaptiveAvgPool2d(1)    # output size = [B, 128, 1, 1]

        # Fully‑connected embedding head
        self.fc1      = nn.Linear(128, 512)
        self.bn_fc    = nn.BatchNorm1d(512)
        self.dropout  = nn.Dropout(0.5)
        self.fc_embed = nn.Linear(512, embedding_dim)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]  (e.g. 256×256)
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1))
        # Now x has shape [B, 128, H/8, W/8]
        x = self.gap(x).view(x.size(0), -1)      # → [B, 128]
        x = F.leaky_relu(self.bn_fc(self.fc1(x)), negative_slope=0.1)
        x = self.dropout(x)
        embedding = self.fc_embed(x)             # → [B, embedding_dim]
        return embedding
