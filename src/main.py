import pandas as pd
import numpy as np
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight


# Step 1: Device Configuration
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Define Model
class DesspressionModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super(DesspressionModel, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Step 3: Clean Data
file_path = 'student_depression_dataset.csv'
data = pd.read_csv(file_path)

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Drop incomplete/uninformative columns
data.drop(columns=['Job Satisfaction', 'Work Pressure', 'id', 'Profession'], inplace=True, errors='ignore')

# Handle missing values
data.replace('?', np.nan, inplace=True)  # Replace '?' with NaN
data.dropna(subset=['Depression'], inplace=True)  # Drop rows where the target is missing
data.fillna(data.median(numeric_only=True), inplace=True)  # Fill missing numerical values with median
data.fillna(data.mode().iloc[0], inplace=True)  # Fill missing categorical values with mode

# Binary columns
binary_columns = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
for col in binary_columns:
    data[col] = data[col].map({'No': 0, 'Yes': 1})

# Encode categorical variables
categorical_columns = ['Gender', 'City', 'Sleep Duration', 'Dietary Habits', 'Degree']  # Update with actual categorical columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True, dtype=int)


# Normalize numerical features
numerical_columns = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']  # Update with actual numerical columns
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Separate features and target
X = data.drop(columns=['Depression'])  # Features
y = data['Depression']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split into training and development sets
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# make a list of all labels
labels = np.unique(y_train)
print(f"Labels: {labels}")

# process the training set
X_train = pt.tensor(X_train.values, requires_grad=False, dtype=pt.float32)
y_train = pt.tensor(y_train.values, requires_grad=False, dtype=pt.long)

# process the testing set
X_test = pt.tensor(X_test.values, requires_grad=False, dtype=pt.float32)
y_test = pt.tensor(y_test.values, requires_grad=False, dtype=pt.long)

# process the development set
X_dev = pt.tensor(X_dev.values, requires_grad=False, dtype=pt.float32)
y_dev = pt.tensor(y_dev.values, requires_grad=False, dtype=pt.long)


# Step 4: Instantiate Model & Move to Device
input_size = X_train.shape[1]  
output_size = len(labels)
hidden_sizes = [32, 16]
model = DesspressionModel(input_size, hidden_sizes, output_size, dropout_rate=0.5).to(device)

# Step 5: Prepare data
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
X_dev = X_dev.to(device)
y_dev = y_dev.to(device)

# Step 6: Define Loss and Optimizer
alpha = 0.001
 
# # Compute class weights
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu()), y=y_train.cpu().numpy())
# class_weights = pt.tensor(class_weights, dtype=pt.float32).to(device)

cost_function = nn.CrossEntropyLoss()
optimizer = pt.optim.Adam(model.parameters(), lr=alpha)  # Adjust weight_decay as needed

from torch.amp import GradScaler, autocast
scaler = GradScaler('cuda' if pt.cuda.is_available() else 'cpu')

# Step 7: Train the Model
batch_size = 128
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 100

cost = []
model.train()
for epoch in range(num_epochs):
    for i, (X_batch, y_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        with autocast('cuda' if pt.cuda.is_available() else 'cpu'):
            output = model(X_batch)
            loss = cost_function(output, y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    cost.append(loss.item())

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Calculate training accuracy
    with pt.no_grad():
        train_outputs = model(X_train)
        _, train_preds = pt.max(train_outputs, 1)
        train_acc = accuracy_score(y_train.cpu(), train_preds.cpu())

        dev_outputs = model(X_dev)
        _, dev_preds = pt.max(dev_outputs, 1)
        dev_acc = accuracy_score(y_dev.cpu(), dev_preds.cpu())

    print(f"Train Accuracy: {train_acc:.4f} | Dev Accuracy: {dev_acc:.4f}")


model.eval()
# Final evaluation on test set
with pt.no_grad():
    test_outputs = model(X_test)
    _, test_preds = pt.max(test_outputs, 1)
    test_acc = accuracy_score(y_test.cpu(), test_preds.cpu())
    print(f"Test Accuracy: {test_acc:.4f}")
