import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworkClassification(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=2, dropout_rate=0.3):
        super(NeuralNetworkClassification, self).__init__()

        # Layer 1: input -> 256
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Layer 2: 256 -> 128
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Layer 3: 128 -> 64
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        
        # Output: 64 -> num_classes
        self.fc_out = nn.Linear(hidden_size // 4, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Weight initialization (Xavier / He)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.fc_out(x)  # Raw logits untuk CrossEntropyLoss
        return x

model = NeuralNetworkClassification(input_size=10, hidden_size=64, num_classes=2)
print("=" * 50)
print("DAFTAR PARAMETER MODEL")
print("=" * 50)

for name, param in model.named_parameters():
    print(f"* Nama : {name}")
    print(f"   Shape: {param.shape}")
    print(f"   Jumlah elemen: {param.numel():,}")
    print(f"   Requires grad: {param.requires_grad}")
    print("-" * 50)