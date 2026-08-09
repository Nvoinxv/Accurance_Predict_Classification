import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworkClassification(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_classes=2, dropout_rate=0.1):
        super(NeuralNetworkClassification, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.BatchNorm1d(hidden_size // 8),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size // 8, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

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