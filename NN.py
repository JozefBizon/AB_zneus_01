import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        #x = self.sigmoid(self.fc3(x))
        x = self.fc3(x)
        return x


def train_model(X_train, y_train, num_epochs=50, learning_rate=0.001):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = SimpleNN(input_size=X_train.shape[1])

    # because classes are unbalanced
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32)

    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    model.eval()
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        #y_pred = model(X_test_tensor)
        y_pred = torch.sigmoid(model(X_test_tensor))
        y_pred_labels = (y_pred > 0.5).float()
        accuracy = (y_pred_labels.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)
    predicted_array = y_pred_labels.cpu().numpy().flatten()
    return accuracy, predicted_array