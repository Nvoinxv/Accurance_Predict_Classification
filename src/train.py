from model import NeuralNetworkClassification
from data_proprecessing import InsuranceClaimPreprocessor
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 1. DATA PREPARATION
# ============================================
processor = InsuranceClaimPreprocessor(
    data_path="/home/nvoinxv/Documents/Classification_Predict_Accurance_Model/Data/Insurance claims data.csv"
)

X_train, X_test, y_train, y_test = processor.run_full_pipeline(
    test_size=0.2, balance_method="smote"
)

# PENTING: Pastikan preprocessing sudah melakukan StandardScaler!
# Kalau belum, tambahkan ini:
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============================================
# 2. MODEL & CONFIG
# ============================================
model = NeuralNetworkClassification(
    input_size=X_train.shape[1],
    hidden_size=256,
    num_classes=2,
    dropout_rate=0.3
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning Rate Scheduler: turunkan LR kalau test loss stuck
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Early Stopping config
early_stop_patience = 15
best_test_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

# List tracking
train_losses, train_accs = [], []
test_losses, test_accs = [], []
num_epochs = 100

# ============================================
# 3. FUNGSI TRAINING
# ============================================
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
        
        # Gradient clipping (stabilkan training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
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
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# ============================================
# 5. MAIN LOOP
# ============================================
print("=" * 70)
print("MULAI TRAINING + VALIDASI (Target: 80%+ dengan Gap Kecil)")
print("=" * 70)

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # Scheduler step berdasarkan test loss
    scheduler.step(test_loss)
    
    # Early Stopping check
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
        print(f"\n>>> Early stopping triggered at epoch {epoch+1} (no improve for {early_stop_patience} epochs)")
        break

# Restore model terbaik
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(">>> Model terbaik telah di-restore.")

print("=" * 70)

# ============================================
# 6. FINAL EVALUATION
# ============================================
final_train_loss, final_train_acc = evaluate(model, train_loader, criterion)
final_test_loss, final_test_acc = evaluate(model, test_loader, criterion)

print(f"\nFINAL RESULTS:")
print(f"  Train -> Loss: {final_train_loss:.4f} | Acc: {final_train_acc:.2f}%")
print(f"  Test  -> Loss: {final_test_loss:.4f} | Acc: {final_test_acc:.2f}%")
print(f"  Gap   -> {abs(final_train_acc - final_test_acc):.2f}%")

# ============================================
# 7. PLOTTING
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot Loss
axes[0].plot(train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
axes[0].plot(test_losses, 'r-', label='Test Loss', linewidth=2, alpha=0.8)
axes[0].axvline(x=np.argmin(test_losses), color='g', linestyle='--', label='Best Model')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss: Train vs Test')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# Plot Accuracy
axes[1].plot(train_accs, 'b-', label='Train Accuracy', linewidth=2, alpha=0.8)
axes[1].plot(test_accs, 'r-', label='Test Accuracy', linewidth=2, alpha=0.8)
axes[1].axvline(x=np.argmin(test_losses), color='g', linestyle='--', label='Best Model')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy: Train vs Test')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('training_improved.png', dpi=150)
plt.show()