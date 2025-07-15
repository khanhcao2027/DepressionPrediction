import torch as pt
import torch.nn as nn
from sklearn.metrics import accuracy_score

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
    
def evaluate_model(model, X, y):
    model.eval()
    with pt.no_grad():
        outputs = model(X)
        _, preds = pt.max(outputs, 1)
        acc = accuracy_score(y.cpu(), preds.cpu())
    return acc

def train_model(model, data_loader, criterion, optimizer, num_epochs=10, Xy_tuple=None):
    X_train, y_train, X_test, y_test, X_dev, y_dev = Xy_tuple
    cost = []
    model.train()
    for epoch in range(num_epochs):
        for X, y in data_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        cost.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {cost[-1]:.4f}')
        # Calculate training accuracy
        train_acc = evaluate_model(model, X_train, y_train)
        dev_acc = evaluate_model(model, X_dev, y_dev)
        print(f"Train Accuracy: {train_acc:.4f} | Dev Accuracy: {dev_acc:.4f}")

