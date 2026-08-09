from model import NeuralNetworkClassification
from data_proprecessing import InsuranceClaimPreprocessor
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# ============================================
# 1. DATA PREPARATION — Pakai RandomOverSampler
# ============================================
processor = InsuranceClaimPreprocessor(
    data_path="/home/nvoinxv/Documents/Classification_Predict_Accurance_Model/Data/Insurance claims data.csv",
    iqr_factor=3.0  # Jangan terlalu agresif hapus outlier
)

# Pakai random oversampling (duplikasi baris asli, aman untuk OHE)
X_train, X_test, y_train, y_test = processor.run_full_pipeline(
    test_size=0.2, balance_method="random"
)

print(f"\nDataset shape:")
print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
print(f"  Class weights: {processor.class_weights}")

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

# BATCH SIZE 128 untuk stabilkan evaluasi
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ============================================
# 2. MODEL & CONFIG
# ============================================
model = NeuralNetworkClassification(
    input_size=X_train.shape[1],
    hidden_size=512,
    num_classes=2,
    dropout_rate=0.15
)

# Class weights tensor
class_weights = torch.tensor(
    [processor.class_weights[0], processor.class_weights[1]],
    dtype=torch.float32
)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# OneCycleLR: naikkan LR di awal, lalu turun perlahan → converge lebih baik
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=200,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # 30% epoch untuk warmup
    anneal_strategy='cos'
)

# Early stopping (monitor TEST LOSS, lebih stabil dari accuracy)
early_stop_patience = 20
best_test_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

train_losses, train_accs = [], []
test_losses, test_accs = [], []
num_epochs = 200

# ============================================
# 3. FUNGSI TRAINING
# ============================================
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # OneCycleLR step per batch
        
        running_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# ============================================
# 4. FUNGSI VALIDATION
# ============================================
def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item() * batch_x.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# ============================================
# 5. MAIN LOOP
# ============================================
print("=" * 70)
print("TRAINING AI")
print("=" * 70)

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # Early stopping berdasarkan TEST LOSS (lebih stabil)
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict().copy()
        print(f"Epoch [{epoch+1:03d}/{num_epochs}]  |  "
              f"TRAIN Loss: {train_loss:.4f} Acc: {train_acc:.2f}%  ||  "
              f"TEST Loss: {test_loss:.4f} Acc: {test_acc:.2f}%  [BEST]")
    else:
        epochs_no_improve += 1
        print(f"Epoch [{epoch+1:03d}/{num_epochs}]  |  "
              f"TRAIN Loss: {train_loss:.4f} Acc: {train_acc:.2f}%  ||  "
              f"TEST Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
    
    if epochs_no_improve >= early_stop_patience:
        print(f"\n>>> Early stopping at epoch {epoch+1} (best test loss: {best_test_loss:.4f})")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

print("=" * 70)

# ============================================
# 6. FINAL EVALUATION + CLASSIFICATION REPORT
# ============================================
final_train_loss, final_train_acc, _, _ = evaluate(model, train_loader, criterion)
final_test_loss, final_test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
gap = abs(final_train_acc - final_test_acc)

print(f"\nFINAL RESULTS:")
print(f"  Train -> Loss: {final_train_loss:.4f} | Acc: {final_train_acc:.2f}%")
print(f"  Test  -> Loss: {final_test_loss:.4f} | Acc: {final_test_acc:.2f}%")
print(f"  Gap   -> {gap:.2f}%")

print(f"\nClassification Report (Test Set):")
print(classification_report(test_labels, test_preds, target_names=['Class 0', 'Class 1'], digits=4))

# ============================================
# 7. PLOTTING
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

best_epoch = np.argmin(test_losses)

axes[0].plot(train_losses, 'b-', label='Train Loss', linewidth=2)
axes[0].plot(test_losses, 'r-', label='Test Loss', linewidth=2)
axes[0].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Ep {best_epoch+1})')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss: Train vs Test'); axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].plot(train_accs, 'b-', label='Train Accuracy', linewidth=2)
axes[1].plot(test_accs, 'r-', label='Test Accuracy', linewidth=2)
axes[1].axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Ep {best_epoch+1})')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy: Train vs Test'); axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('training_fixed.png', dpi=150)
plt.show()