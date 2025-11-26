import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
'''
Most information taken from 
Playlist:
https://www.youtube.com/watch?v=kY14KfZQ1TI&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1
https://setosa.io/ev/image-kernels/
'''

# global variables
batch_size = 10
learning_rate = 0.001
seed=0
epochs = 5
SGD_momentum=0.9

transform = transforms.ToTensor()

# data download/load
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# cnn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.lin1 = nn.Linear(5 * 5 * 16, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)

    def forward(self, data):
        data = F.relu(self.conv1(data))
        data = F.max_pool2d(data, 2, 2)
        data = F.relu(self.conv2(data))
        data = F.max_pool2d(data, 2, 2)
        data = data.view(-1, 16 * 5 * 5)  # Flatten
        data = F.relu(self.lin1(data))
        data = F.relu(self.lin2(data))
        data = self.lin3(data)
        return F.log_softmax(data, dim=1)

torch.manual_seed(seed)
cnn = CNN()
all_preds = []
all_labels = []

criterion = nn.CrossEntropyLoss()
opt=int(input("Which optimizer to use? (For optimizer)-type:\n"
              f"SGD - 1\nSGD with momentum {SGD_momentum} - 2\nADAM - N>2\n"))
if opt==1:
  optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
  print("Training with SGD optimizer.\n")
elif opt==2:
  optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate,momentum=SGD_momentum)
  print("Training with SGD optimizer with momentum.\n")
else:
  optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
  print("Training with ADAM optimizer.\n")

train_losses = []
test_losses = []
train_correct = []
test_correct = []

start_time = time.time()


for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # training
    cnn.train()
    for batch_num, (X_train, y_train) in enumerate(train_loader):
        batch_num += 1
        y_pred = cnn(X_train)
        loss = criterion(y_pred, y_train)

        # weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        # prints status every one/tenth of each epoch
        if batch_num % ((len(train_data)/batch_size)/10) == 0:
            print(f'Epoch: {i+1}  Batch: {batch_num}  Loss: {loss}')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # testing in each epoch w no_grad(), so the torch won't change
    cnn.eval()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_val = cnn(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

    train_accuracy = trn_corr / len(train_data) * 100
    test_accuracy = tst_corr / len(test_data) * 100
    print(f'Epoch {i+1} - Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

total_time = (time.time() - start_time) / 60
print(f'Training took: {total_time:.2f} minutes')

# loss graph # from CHATGPT
plt.plot([l.item() for l in train_losses], label="Training Loss")
plt.plot([l.item() for l in test_losses], label="Validation Loss")
plt.title("Loss at Epoch")
plt.legend()
plt.show()

# acc graph # from CHATGPT
plt.plot([t / len(train_data) * 100 for t in train_correct], label="Training Accuracy")
plt.plot([t / len(test_data) * 100 for t in test_correct], label="Validation Accuracy")
plt.title("Accuracy at the end of each Epoch")
plt.legend()
plt.show()

# run whole test
test_load_everything = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_everything:
        y_val = cnn(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())

accuracy = correct / len(test_data) * 100
print(f'Final Test Accuracy: {accuracy:.2f}%')

# graphs and confusion matrix  # from CHATGPT
with torch.no_grad():
    for X_test, y_test in DataLoader(test_data, batch_size=10, shuffle=True):
        y_val = cnn(X_test)
        predicted = torch.max(y_val, 1)[1]
        print(f'Predicted: {predicted.numpy()}')
        print(f'Actual: {y_test.numpy()}')
        plt.imshow(make_grid(X_test, nrow=10).permute(1, 2, 0))
        plt.show()
        break

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()