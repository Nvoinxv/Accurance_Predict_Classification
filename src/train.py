from model import NeuralNetworkClassification
from data_proprecessing import InsuranceClaimPreprocessor
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt

processor = InsuranceClaimPreprocessor(
    data_path="/home/nvoinxv/Documents/Classification_Predict_Accurance_Model/Data/Insurance claims data.csv"
)

X_train, X_test, y_train, y_test = processor.run_full_pipeline(
    test_size=0.2, balance_method="smote"
)

# Langsung konversi ke tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.long)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("=" * 50)
print("Bebebrapa sample data train_loader")

# Get the first batch
for batch_idx, (inputs, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  First 5 inputs:\n{inputs[:5]}")
    print(f"  First 5 labels: {labels[:5]}")
    break  # Only show the first batch

print("=" * 50)

model = NeuralNetworkClassification(input_size=X_train.shape[1],
                                    num_classes=2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Menyimpan metrik latihan AI
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc