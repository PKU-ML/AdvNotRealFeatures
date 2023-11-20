import os
import json
import torch, torchvision
from datetime import datetime
from functools import partial
from ddpm_torch import *
from torch.optim import Adam, lr_scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing import errors
import numpy as np
from torch.utils.data import DataLoader
from dataset import *

def train(rank=0, args=None, temp_dir=""):

    def logger(msg, **kwargs):
        print(msg, **kwargs)

    dataset = args.dataset
    if 'cifar10' in dataset:
        dataset = 'cifar10'
    in_channels = DATASET_INFO[dataset]["channels"]
    image_res = DATASET_INFO[dataset]["resolution"]
    image_shape = (in_channels, ) + image_res

    # set seed for all rngs
    seed = args.seed
    seed_all(seed)

    configs_path = os.path.join(args.config_dir, dataset + ".json")
    with open(configs_path, "r") as f:
        configs = json.load(f)

    # train parameters
    gettr = partial(get_param, configs_1=configs.get("train", {}), configs_2=args)
    train_configs = ConfigDict(**{
        k: gettr(k)
        for k in ("batch_size", "beta1", "beta2", "lr", "epochs", "grad_norm", "warmup")
    })
    train_configs.batch_size //= args.num_accum
    train_device = torch.device(args.train_device)
    eval_device = torch.device(args.eval_device)

    # diffusion parameters
    getdif = partial(get_param, configs_1=configs.get("diffusion", {}), configs_2=args)
    diffusion_configs = ConfigDict(**{
        k: getdif(k)
        for k in (
            "beta_schedule",
            "beta_start",
            "beta_end",
            "timesteps",
            "model_mean_type",
            "model_var_type",
            "loss_type"
        )})

    betas = get_beta_schedule(
        diffusion_configs.beta_schedule, beta_start=diffusion_configs.beta_start,
        beta_end=diffusion_configs.beta_end, timesteps=diffusion_configs.timesteps)
    diffusion = GaussianDiffusion(betas=betas, **diffusion_configs)

    # denoise parameters
    out_channels = 2 * in_channels if diffusion_configs.model_var_type == "learned" else in_channels
    model_configs = configs["denoise"]
    block_size = model_configs.pop("block_size", args.block_size)
    model_configs["in_channels"] = in_channels * block_size ** 2
    model_configs["out_channels"] = out_channels * block_size ** 2
    _model = UNet(**model_configs)

    if block_size > 1:
        pre_transform = torch.nn.PixelUnshuffle(block_size)  # space-to-depth
        post_transform = torch.nn.PixelShuffle(block_size)  # depth-to-space
        _model = ModelWrapper(_model, pre_transform, post_transform)

    model = _model.to(train_device)

    logger(f"Dataset: {dataset}")
    logger(
        f"Effective batch-size is {train_configs.batch_size} * {args.num_accum}"
        f"= {train_configs.batch_size * args.num_accum}.")

    optimizer = Adam(model.parameters(), lr=train_configs.lr, betas=(train_configs.beta1, train_configs.beta2))
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / train_configs.warmup, 1.0)
    ) if train_configs.warmup > 0 else None

    split = "all" if dataset == "celeba" else "train"
    num_workers = args.num_workers
    trainloader, sampler = get_dataloader(
        dataset, batch_size=train_configs.batch_size, split=split, val_size=0., random_seed=seed,
        root='../data', drop_last=True, pin_memory=True, num_workers=num_workers
    )  

    if 'robust' in args.dataset:
        trainloader.dataset = Crafted_CIFAR10_Training_Set(trainloader.dataset.transform, args.dataset)

    chkpt_dir = args.chkpt_dir
    chkpt_path = os.path.join(chkpt_dir, args.chkpt_name or f"ddpm_{dataset}.pt")
    chkpt_intv = args.chkpt_intv
    image_dir = os.path.join(args.image_dir, f"{dataset}")
    image_intv = args.image_intv
    num_save_images = args.num_save_images
    
    model_configs["block_size"] = block_size
    hps = {
        "dataset": dataset,
        "seed": seed,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
        "num_accum": args.num_accum,
        "train": train_configs,
        "denoise": model_configs,
        "diffusion": diffusion_configs
    }
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")

    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    # keep a record of hyperparameter settings used for this experiment run
    with open(os.path.join(chkpt_dir, f"exp_{timestamp}.info"), "w") as f:
        json.dump(hps, f, indent=2)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        diffusion=diffusion,
        epochs=train_configs.epochs,
        trainloader=trainloader,
        sampler=sampler,
        scheduler=scheduler,
        num_accum=args.num_accum,
        use_ema=args.use_ema,
        grad_norm=train_configs.grad_norm,
        shape=image_shape,
        device=train_device,
        chkpt_intv=chkpt_intv,
        image_intv=image_intv,
        num_save_images=num_save_images,
        ema_decay=args.ema_decay,
        rank=rank,
        dry_run=args.dry_run
    )
    evaluator = Evaluator(dataset=dataset, device=eval_device) if args.eval else None
    
    # use cudnn benchmarking algorithm to select the best conv algorithm
    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa
        logger(f"cuDNN benchmark: ON")

    logger("Training starts...", flush=True)
    trainer.train(evaluator, chkpt_path=chkpt_path, image_dir=image_dir)
    
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
    cls_model = Diffusion_Classifier(diffusion, model)
    for param in cls_model.parameters():
        param.requires_grad = False
    for param in cls_model.linear.parameters():
        param.requires_grad = True
    
    import math
    optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for e in range(args.total_epoch):
        cls_model.train()
        for img, label in iter(train_loader):
            img, label = img.cuda(), label.cuda()
            
            logits = cls_model(img)
            loss = loss_fn(logits, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.linear.parameters(), 5.0)
            optimizer.step()
            
        cls_model.eval()
        total_correct = 0
        with torch.no_grad():
            for img, label in iter(test_loader):
                img, label = img.cuda(), label.cuda()
                logits = cls_model(img)
                total_correct += (torch.argmax(logits, axis=1) == label).sum().detach().item()
        test_acc = total_correct/10000
        scheduler.step()
        print(f'SL, epoch {e}, Dataset = {args.dataset}, Acc = {test_acc}')
        scheduler.step()
    
    return test_acc


class Diffusion_Classifier(torch.nn.Module):
    def __init__(self, diffusion:GaussianDiffusion, encoder:UNet):
        super().__init__()
        self.diffsuion = diffusion
        self.encoder = encoder
        self.t = torch.tensor(11, dtype=torch.long).cuda()
        self.pooling = torch.nn.AdaptiveAvgPool2d((16, 16))
        self.linear = torch.nn.Linear(65536, 10)
    
    def forward(self, x):
        x_t = self.diffsuion.q_sample(x, self.t, noise=None)
        try:
            embedding = self.encoder.encoder_forward(x_t, self.t)
            embedding = self.pooling(embedding)
        except:
            embedding = self.encoder.module.encoder_forward(x_t, self.t)
            embedding = self.pooling(embedding)
        return self.linear(embedding.reshape((embedding.shape[0], -1)))

@errors.record
def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "robust-cifar10", "non-robust-cifar10"], default="cifar10")
    parser.add_argument("--data-path", default="../data", type=str, help="root directory of datasets")
    parser.add_argument("--epochs", default=50, type=int, help="total number of training epochs")
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta_1 in Adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta_2 in Adam")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-accum", default=1, type=int, help="number of mini-batches before an update")
    parser.add_argument("--block-size", default=1, type=int, help="block size used for pixel shuffle")
    parser.add_argument("--timesteps", default=1000, type=int, help="number of diffusion steps")
    parser.add_argument("--beta-schedule", choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
    parser.add_argument("--beta-start", default=0.0001, type=float)
    parser.add_argument("--beta-end", default=0.02, type=float)
    parser.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="eps", type=str)
    parser.add_argument("--model-var-type", choices=["learned", "fixed-small", "fixed-large"], default="fixed-large", type=str)  # noqa
    parser.add_argument("--loss-type", choices=["kl", "mse"], default="mse", type=str)
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers for data loading")
    parser.add_argument("--train-device", default="cuda:0", type=str)
    parser.add_argument("--eval-device", default="cuda:0", type=str)
    parser.add_argument("--image-dir", default="./images/train", type=str)
    parser.add_argument("--image-intv", default=1, type=int)
    parser.add_argument("--num-save-images", default=64, type=int, help="number of images to generate & save")
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-name", default="", type=str)
    parser.add_argument("--chkpt-intv", default=500, type=int, help="frequency of saving a checkpoint")
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--resume", action="store_true", help="to resume training from a checkpoint")
    parser.add_argument("--chkpt-path", default="", type=str, help="checkpoint path used to resume training")
    parser.add_argument("--eval", action="store_true", help="whether to evaluate fid during training")
    parser.add_argument("--use-ema", action="store_true", help="whether to use exponential moving average")
    parser.add_argument("--ema-decay", default=0.9999, type=float, help="decay factor of ema")
    parser.add_argument("--rigid-launch", action="store_true", help="whether to use torch multiprocessing spawn")
    parser.add_argument("--num-gpus", default=1, type=int, help="number of gpus for distributed training")
    parser.add_argument("--dry-run", action="store_true", help="test-run till the first model update completes")
    args = parser.parse_args()

    print(f'Start training DM on dataset {args.dataset}')
    test_acc = train(args=args)
    
    if not os.path.exists('./log.txt'):
        open('./log.txt', 'w')
    with open('log.txt', 'a') as f:
        f.write(f'Diffusion Model (DDPM), dataset = {args.datset}, acc = {test_acc}\n')
    
if __name__ == "__main__":
    main()