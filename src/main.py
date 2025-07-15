import pandas as pd
import numpy as np
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_class_weight

from model import DesspressionModel, train_model
from data.data_utils import clean_depression_data, preprocess_data  # <-- import functions
import cuda

# Step 1: Device Configuration
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 3: Load and Preprocess Data
X, y = clean_depression_data()
Xy_tuple = preprocess_data(X, y)
X_train, y_train, X_test, y_test, X_dev, y_dev = Xy_tuple

# make a list of all labels
labels = np.unique(y_train.cpu())
print(f"Labels: {labels}")

# # process the training set
# X_train = pt.tensor(X_train.values, requires_grad=False, dtype=pt.float32)
# y_train = pt.tensor(y_train.values, requires_grad=False, dtype=pt.long)

# # process the testing set
# X_test = pt.tensor(X_test.values, requires_grad=False, dtype=pt.float32)
# y_test = pt.tensor(y_test.values, requires_grad=False, dtype=pt.long)

# # process the development set
# X_dev = pt.tensor(X_dev.values, requires_grad=False, dtype=pt.float32)
# y_dev = pt.tensor(y_dev.values, requires_grad=False, dtype=pt.long)


# Step 4: Instantiate Model & Move to Device
input_size = X_train.shape[1]  
output_size = len(labels)
hidden_sizes = [32, 16]
model = DesspressionModel(input_size, hidden_sizes, output_size, dropout_rate=0.5).to(device)

# Step 5: Load Data to Device
Xy_tuple = cuda.to_device(Xy_tuple, cuda.current_device)

# Step 6: Define Loss and Optimizer
alpha = 0.001

cost_function = nn.CrossEntropyLoss()
optimizer = pt.optim.Adam(model.parameters(), lr=alpha)  # Adjust weight_decay as needed

# Step 7: Train the Model
batch_size = 128
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 100
train_model(model, dataloader, cost_function, optimizer, num_epochs)


model.eval()
# Final evaluation on test set
with pt.no_grad():
    test_outputs = model(X_test)
    _, test_preds = pt.max(test_outputs, 1)
    test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    print(f"Test Accuracy: {test_acc:.4f}")
