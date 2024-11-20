# Imports for pytorch (maybe redundant but its ok)
import numpy as np
import torch
import torchvision
from torch import nn
import matplotlib
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

# Creating the datasets
transform = torchvision.transforms.ToTensor() # feel free to modify this as you see fit.

training_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

validation_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

# see if we can use GPU instead of CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)

# set neurons per layer
neurons_per_layer = 420

# define our neural network
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()

    # formats into multi layer perceptron (fully connected)
    # 2 hidden layers, variable neurons per later (we can mess with later)
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28*28, neurons_per_layer),
      nn.ReLU(),
      nn.Linear(neurons_per_layer, neurons_per_layer),
      nn.ReLU(),
      nn.Linear(neurons_per_layer, 10)
    )

  # forward propogation
  def forward(self, x):
    x = self.flatten(x)
    return self.linear_relu_stack(x)

# some useful variables for later on
epochs = 10
batch_size = 30
learning_rate = 0.042
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []
epochs_x = []

model = NeuralNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# we will use cross entropy loss, so we wont be adding softmax layer at the end
loss_function = nn.CrossEntropyLoss()
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

# putting everything together (reusing some code from example)
# added comments to show understanding

model.train()
for epoch in range(epochs):
  training_losses = []
  epochs_x.append(epoch + 1)

  # for all batches in dataset
  num_correct = 0
  for x, y in tqdm(training_dataloader, unit="batch"):
    x, y = x.to(device), y.to(device) # removed .float() bc it was useless i think
    optimizer.zero_grad() # Remove the gradients from the previous step
    pred = model(x)
    loss = loss_function(pred, y)
    loss.backward()
    optimizer.step()
    training_losses.append(loss.item())
    num_correct += torch.sum(torch.argmax(pred, dim=1) == y).item()
  training_loss.append(np.mean(training_losses))
  training_accuracy.append(num_correct / len(training_data))
  print("Finished Epoch", epoch + 1, ', Calculating accuracy and loss:')

  # obtaining training and validation accuracy
  # put model in evaluation mode
  model.eval()

  with torch.no_grad():
    # training accuracy
    # num_correct = 0
    # for x, y in tqdm(training_dataloader, unit="batch"):
    #   x, y = x.to(device), y.to(device)
    #   pred = model(x)
    #   num_correct += torch.sum(torch.argmax(pred, dim=1) == y).item()
    # training_accuracy.append(num_correct / len(training_data))

    # validation loss and accuracy
    validation_losses = []
    num_correct = 0
    for x, y in tqdm(validation_dataloader, unit="batch"):
      x, y = x.to(device), y.to(device)
      pred = model(x)
      loss = loss_function(pred, y)
      validation_losses.append(loss.item())
      num_correct += torch.sum(torch.argmax(pred, dim=1) == y).item()
    validation_loss.append(np.mean(validation_losses))
    validation_accuracy.append(num_correct / len(validation_data))

    # output
    print("Epoch", epoch + 1, ": TL:", training_loss[-1],
          "TA:", training_accuracy[-1],
          "VL:", validation_loss[-1],
          "VA:", validation_accuracy[-1])

    # put model back in training mode
    model.train()


# plotting the data
# training and validation loss
plt.figure()
plt.plot(epochs_x, training_loss, label="Training Loss")
plt.plot(epochs_x, validation_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss vs Epoch")
plt.legend()
plt.show()

# training and validation accuracy
plt.figure()
plt.plot(epochs_x, training_accuracy, label="Training Accuracy")
plt.plot(epochs_x, validation_accuracy, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy vs Epoch")
plt.legend()
plt.show()