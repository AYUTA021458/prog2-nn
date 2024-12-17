import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models



ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)
ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

train_loader = dataloader_train
test_loader = dataloader_test

for image_batch, label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break

model = models.MyModel()

acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100}%')

model = models.MyModel()

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 0.003
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 10  
learning_rate = 0.001  

model = models.MyModel()  
loss_fn = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

n_epochs = 5

for k in range(n_epochs):
    print(f'epochs {k+1}/{n_epochs}', end=':', flush=True)

    loss_train = models.train(model, dataloader_train, loss_fn, optimizer)
    print(f'train loss: {loss_train}')

    acc_test = models.test_accuracy(model, dataloader_test)
    print(f'test accuracy: {acc_test*100:.2f}%')

def test(model, dataloader, loss_fn):
    loss_total = 0.0

    model.eval()
    for image_batch, label_batch in dataloader:
        with torch.no_grad():
            logits_batch = model(image_batch)

        loss = loss_fn(logits_batch, label_batch)
        loss_total += loss.item()

    return loss_total / len(dataloader)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Over Epochs")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Over Epochs")
plt.show()
