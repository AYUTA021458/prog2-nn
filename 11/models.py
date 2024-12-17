import time

import matplotlib.pyplot as plt

import torch 
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits


def test_accuracy(model, dataloader):
    n_corrects = 0

    model.eval()
    for image_batch, label_batch in dataloader:
        with torch.no_grad():
            logits_batch = model(image_batch)
        
        predict_batch = logits_batch.argmax(dim=1)
        n_corrects += (label_batch == predict_batch).sum().item()

    accuracy = n_corrects / len(dataloader.dataset)
    return accuracy

def train(model, dataloader, loss_fn, optimizer):
    model.train()
    for image_batch, label_batch in dataloader:
        logits_batch = model(image_batch)

        loss = loss_fn(logits_batch, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

batch_size = 64
ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)

dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
)
model = models.MyModel()
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 0.003
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')

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
        with torch.np_grad():
            logits_batch = model(image_batch)

        loss = loss_fn(logits_batch, label_batch)
        loss_total += loss.item()

    return loss_total / len(dataloader)