import argparse, math, torch, os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--max_device_batch_size', type=int, default=512)
parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--total_epoch', type=int, default=1000)
parser.add_argument('--warmup_epoch', type=int, default=200)
parser.add_argument('--mask-ratio', type=float, default=0.75)
parser.add_argument('--dataset', type=str, choices=['cifar10','robust-cifar10', 'non-robust-cifar10'],  default='cifar10')
args = parser.parse_args()

print(f'Start training MIM on dataset {args.dataset}')

batch_size = args.batch_size
load_batch_size = min(args.max_device_batch_size, batch_size)
steps_per_update = batch_size // load_batch_size

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
train_set = datasets.CIFAR10('../data', train=True,  transform=transform_train, download=True)
if 'robust' in args.dataset:
    train_set = Crafted_CIFAR10_Training_Set(transform_train, args.dataset)
dataloader = DataLoader(train_set, load_batch_size, shuffle=True, num_workers=8)

model = MAE_ViT(image_size=32, patch_size=8, emb_dim=384, encoder_layer=12, encoder_head=6, decoder_layer=6, decoder_head=6, mask_ratio=args.mask_ratio).cuda()
optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

step_count = 0
optim.zero_grad()
for e in tqdm(range(args.total_epoch)):
    model.train()
    for img, label in tqdm(iter(dataloader)):
        step_count += 1
        img = img.cuda()
        predicted_img, mask = model(img)
        loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
        loss.backward()
        if step_count % steps_per_update == 0:
            optim.step()
            optim.zero_grad()
    lr_scheduler.step()

# Do linear probing
model = ViT_Classifier(model.encoder).cuda()
loss_fn = torch.nn.CrossEntropyLoss()

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor()
])
train_set = datasets.CIFAR10('../data', train=True, transform=transform_train, download=True)
test_set = datasets.CIFAR10('../data', train=False, transform=transform_test, download=True)
if 'robust' in args.dataset:
    train_set = Crafted_CIFAR10_Training_Set(transform_train, args.dataset)
train_loader = DataLoader(train_set, shuffle=False, batch_size=512, num_workers=8)
test_loader = DataLoader(test_set, shuffle=False, batch_size=512, num_workers=8)

for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

optimizer = torch.optim.SGD(model.head.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

for e in range(50):
    model.train()
    losses = []
    acces = []
    for img, label in tqdm(iter(train_loader)):
        img, label = img.cuda(), label.cuda()
        logits = model(img)
        loss = loss_fn(logits, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 5.0)
        optimizer.step()
    scheduler.step()
    
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for img, label in tqdm(iter(test_loader)):
            img = img.cuda()
            label = label.cuda()
            logits = model(img)
            total_correct += (torch.argmax(logits, axis=1) == label).sum().detach().item()
    test_acc = total_correct/10000
    scheduler.step()
    print(f'MAE, epoch {e}, Dataset = {args.dataset}, Acc = {test_acc}')
    
# Logging
if not os.path.exists('./log.txt'):
    open('./log.txt', 'w')
with open('./log.txt', 'a') as f:
    f.write(f"Masked Image Modeling (MAE), Dataset = {args.dataset}, Acc = {test_acc}\n")