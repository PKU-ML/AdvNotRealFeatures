import torch, os, torchvision
import torch.backends.cudnn as cudnn
from resnet import ResNetSimCLR
from argparse import ArgumentParser
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from dataset import *

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'robust-cifar10', 'non-robust-cifar10'])
parser.add_argument('--data-path', type=str)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--n_views', type=int, default=2)
parser.add_argument('--temperature', type=float, default=0.007)

args = parser.parse_args()

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=False)
        
        for epoch_counter in range(self.args.epochs):
            for images, __ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)
                with autocast(enabled=False):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                
            # warmup for the first 10 epochsd
            if epoch_counter >= 10:
                self.scheduler.step()

def main():
    # check if gpu training is available
    args.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True

    dataset = ContrastiveLearningDataset('../data').get_dataset('cifar10', 2)
    if 'robust' in args.dataset:
        dataset = Crafted_CIFAR10_Training_Set(dataset.transform, args.dataset)
    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, drop_last=True)
    
    model = ResNetSimCLR(out_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)
    return simclr

def train_classifier(backbone):
    # Data
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    train_set = torchvision.datasets.CIFAR10('../data', train=True, transform=transform_train, download=True)
    test_set = torchvision.datasets.CIFAR10('../data', train=False, transform=transform_test, download=True)
    if 'robust' in args.dataset:
        train_set = Crafted_CIFAR10_Training_Set(transform_train, args.dataset)
    train_loader = DataLoader(train_set, shuffle=False, batch_size=512, num_workers=8)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=512, num_workers=8)

    # Change to probing head
    backbone.linear = torch.nn.Linear(backbone.linear.in_features, 10)
    for param in backbone.parameters():
        param.requires_grad = False
    for param in backbone.linear.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(backbone.linear.parameters(),lr=0.1, momentum=0.9, weight_decay=1e-5)    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.0)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(0, 30):
        for data, labels in train_loader:
            data = data.cuda()
            labels = labels.cuda()
            
            logits = backbone(data)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.linear.parameters(), 5.0)
            optimizer.step()
        
        correct = 0
        for data, labels in test_loader:
            data = data.cuda()
            labels = labels.cuda()
            logits = backbone(data)
            
        scheduler.step()
        test_acc = correct/10000
        print(f'CL, epoch {epoch}, Dataset = {args.dataset}, Acc = {test_acc}')
    
    return test_acc
    
if __name__ == "__main__":
    print(f'Start training CL on dataset {args.dataset}')
    backbone = main()
    test_acc = train_classifier(backbone)
    if not os.path.exists('./log.txt'):
        open('./log.txt', 'w')
    with open('log.txt', 'a') as f:
        f.write(f"Contrastive Learning (SimCLR), Dataset = {args.dataset}, Acc = {test_acc}\n")
