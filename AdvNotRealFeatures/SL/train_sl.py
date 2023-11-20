import torch, os
import numpy as np
from torchvision.datasets import CIFAR10
from resnet import *
from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import *

parser = ArgumentParser()
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'robust-cifar10', 'non-robust-cifar10'])
parser.add_argument('--save', type=str, default='')
parser.add_argument('--model', type=str, default='resnet50')
args = parser.parse_args()

print(f'Start training SL on dataset {args.dataset}')

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor()
])
train_set = CIFAR10('../data', train=True, transform=transform_train, download=True)
test_set = CIFAR10('../data', train=False, transform=transform_test, download=True)
if 'robust' in args.dataset:
    train_set = Crafted_CIFAR10_Training_Set(None, args.dataset)
train_loader = DataLoader(train_set, shuffle=False, batch_size=512, num_workers=8)
test_loader = DataLoader(test_set, shuffle=False, batch_size=512, num_workers=8)
    
# Model
if args.model == 'resnet18':
    model = ResNet18().cuda()
elif args.model == 'resnet50':
    model = ResNet18().cuda()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)

# Training
Acc = []
for i in range(0, 30):
    model.train()
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        logits = model(data)
        loss = loss_function(logits, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            logits = model(data)
            total_correct += (torch.argmax(logits, axis=1) == label).sum().detach().item()
    test_acc = total_correct/10000
    scheduler.step()
    print(f'SL, epoch {i}, Dataset = {args.dataset}, Acc = {test_acc}')

# Logging
if not os.path.exists('./log.txt'):
    open('./log.txt', 'w')
with open('./log.txt', 'a') as f:
    f.write(f"Supervised Learning (SL), Dataset = {args.dataset}, Acc = {test_acc}\n")
    
if args.save:
    torch.save(model.state_dict(), args.save)