import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworkClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetworkClassification, self).__init__()

        self.fcl = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fcl(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.sigmoid(self.fc3(x))

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