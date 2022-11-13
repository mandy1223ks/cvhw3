import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms, models, datasets
from tqdm import tqdm
import pdb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# %matplotlib inline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using {device} device')

def set_all_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def build_model(N, PRETRAIN):
    num_model = 'resnet'+str(N)
    model = torch.hub.load('pytorch/vision:v0.10.0', num_model, pretrained=PRETRAIN).to(device)
    if N == 50:
        model.fc = torch.nn.Linear(2048, 10)
    else:
        model.fc = torch.nn.Linear(512, 10)
    return model    
    
def train(dataloader, model, loss_fn, optimizer):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    epoch_loss = 0
    correct = 0

    model.to(device).train()

    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()

    avg_epoch_loss = epoch_loss / num_batches
    avg_acc = correct / size

    return avg_epoch_loss, avg_acc

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    epoch_loss = 0
    correct = 0

    model.eval()

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            epoch_loss += loss_fn(pred, y).item()
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

    avg_epoch_loss = epoch_loss / num_batches
    avg_acc = correct / size

    return avg_epoch_loss, avg_acc

def train_test_loop(train_dataloader, valid_dataloader, N, PRETRAIN):
    model = build_model(N, PRETRAIN)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 200
    acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_acc = test(valid_dataloader, model, loss_fn)
        if test_acc>acc:
            acc = test_acc
            i = epoch
        print(f"Epoch {epoch + 1:2d}: Loss = {train_loss:.4f} Acc = {train_acc:.2f} Test_Loss = {test_loss:.4f} Test_Acc = {test_acc:.2f}")
    print("Done!")
    print("the acc of test:{}".format(acc))
    print("the epoch:{}".format(i))
    return acc


# PRETRAIN = "IMAGENET1K V1"    
PRETRAIN = False

set_all_seed(123)
batch_size = 256

train_transform = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
valid_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

sixteenth_train_sampler = RandomSampler(train_dataset, num_samples=len(train_dataset)//16, replacement=True)
half_train_sampler = RandomSampler(train_dataset, num_samples=len(train_dataset)//2, replacement=True)

sixteenth_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sixteenth_train_sampler)
half_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=half_train_sampler)

acc_resnet50_1_nw = train_test_loop(train_dataloader, valid_dataloader, 50, PRETRAIN)
acc_resnet18_1_nw = train_test_loop(train_dataloader, valid_dataloader, 18, PRETRAIN)
acc_resnet50_05_nw = train_test_loop(half_train_dataloader, valid_dataloader, 50, PRETRAIN)
acc_resnet18_05_nw = train_test_loop(half_train_dataloader, valid_dataloader, 18, PRETRAIN)
acc_resnet50_016_nw = train_test_loop(half_train_dataloader, valid_dataloader, 50, PRETRAIN)
acc_resnet18_016_nw = train_test_loop(half_train_dataloader, valid_dataloader, 18, PRETRAIN)

x = np.array([1/16, 1/2, 1])
y_s = np.array([acc_resnet18_016_nw, acc_resnet18_05_nw, acc_resnet18_1_nw])
y_b = np.array([acc_resnet50_016_nw, acc_resnet50_05_nw, acc_resnet50_1_nw])
plt.xlabel('Dataset Size')
plt.ylabel('Accuracy')
plt.title('Dataset Size vs Accuracy')
plt.plot(x, y_s, '-o',color='b', label='Small Model')
plt.plot(x, y_b, '-o',color='r', label='Big Model')
# plt.grid()
plt.legend()
plt.show()
plt.savefig('result_4.png')