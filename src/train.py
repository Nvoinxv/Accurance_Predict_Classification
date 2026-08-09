from model import NeuralNetworkClassification
from data_proprecessing import InsuranceClaimPreprocessor
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split

processor = InsuranceClaimPreprocessor(
    data_path="/home/nvoinxv/Documents/Classification_Predict_Accurance_Model/Data/Insurance claims data.csv"
)

x, y = processor.run()
x, y = TensorDataset(x, y)
x, y = DataLoader(x, y, batch_size=10, shuffle=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("=" * 50)
print("Bebebrapa sample data X_train dan X_test")
print(f"X_train: {x_train.head(5)}")
print(f"X_test: {x_test.head(5)}")
print("=" * 50)

print("=" * 50)
print("Bebebrapa sample data y_train dan y_test")
print(f"y_train: {y_train.head(5)}")
print(f"y_test: {y_test.head(5)}")
print("=" * 50)

