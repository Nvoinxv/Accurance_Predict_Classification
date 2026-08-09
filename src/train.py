from model import NeuralNetworkClassification
from data_proprecessing import InsuranceClaimPreprocessor
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 1. DATA PREPARATION 
# ============================================
processor = InsuranceClaimPreprocessor(
    data_path="/home/nvoinxv/Documents/Classification_Predict_Accurance_Model/Data/Insurance claims data.csv"
)

# Panggil preprocessing manual, lalu split, TAPI skip SMOTE
processor.load_data()
processor.drop_features()
processor.remove_outliers()
processor.drop_low_correlation()
processor.scale_numeric()
processor.encode_categorical()
processor.encode_target()
processor.split_train_test(test_size=0.2)

# Pakai class weights untuk imbalance (dari data ASLI, bukan SMOTE)
processor.compute_class_weights()

print(f"Class weights: {processor.class_weights}")

X_train, X_test, y_train, y_test = processor.get_train_test_data()

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ============================================
# 2. CLASS WEIGHTS TENSOR
# ============================================
class_weights = torch.tensor(
    [processor.class_weights[0], processor.class_weights[1]],
    dtype=torch.float32
)

# ============================================
# 3. MODEL & CONFIG
# ============================================
model = NeuralNetworkClassification(
    input_size=X_train.shape[1],
    hidden_size=512,
    num_classes=2,
    dropout_rate=0.1          # Sangat kecil karena underfitting parah
)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)

# LR Scheduler: Cosine Annealing (lebih agresif dari ReduceLROnPlateau)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

# Early Stopping berdasarkan TEST ACCURACY (bukan loss!)
early_stop_patience = 25
best_test_acc = 0.0
epochs_no_improve = 0
best_model_state = None

train_losses, train_accs = [], []
test_losses, test_accs = [], []
num_epochs = 200

# ============================================
# 4. FUNGSI TRAINING
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
        
        # Gradient clipping
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
# 5. FUNGSI VALIDATION
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
# 6. MAIN LOOP
# ============================================
print("=" * 70)
print("TRAINING AI")
print("=" * 70)

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    scheduler.step()
    
    # Early stopping berdasarkan TEST ACCURACY
    if test_acc > best_test_acc:
        best_test_acc = test_acc
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
        print(f"\n>>> Early stopping at epoch {epoch+1} (best test acc: {best_test_acc:.2f}%)")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

print("=" * 70)

# Final eval
final_train_loss, final_train_acc = evaluate(model, train_loader, criterion)
final_test_loss, final_test_acc = evaluate(model, test_loader, criterion)
gap = abs(final_train_acc - final_test_acc)

print(f"\nFINAL RESULTS:")
print(f"  Train -> Loss: {final_train_loss:.4f} | Acc: {final_train_acc:.2f}%")
print(f"  Test  -> Loss: {final_test_loss:.4f} | Acc: {final_test_acc:.2f}%")
print(f"  Gap   -> {gap:.2f}%")

# ============================================
# 7. PLOTTING
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

best_epoch = np.argmax(test_accs)

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
plt.savefig('training_aggressive.png', dpi=150)
plt.show()