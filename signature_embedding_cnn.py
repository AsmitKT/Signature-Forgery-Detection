import torch
import torch.nn as nn
import torch.nn.functional as F

class SignatureEmbeddingCNN(nn.Module):
    """
    CNN that maps a 1×256×256 signature image → a D-dim embedding vector.
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        # Convolutional blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)

        # Fully‐connected heads
        # 256→128→64→32 spatially; 128 channels → 128×32×32 features
        self.fc1      = nn.Linear(128 * 32 * 32, 512)
        self.dropout  = nn.Dropout(0.3)
        self.fc_embed = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 256, 256]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # → [B,32,128,128]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # → [B,64,64,64]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # → [B,128,32,32]
        x = x.view(x.size(0), -1)                       # → [B, 128*32*32]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc_embed(x)                         # → [B, embedding_dim]
