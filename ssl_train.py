import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, OrderedDict
import torchvision as tv
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch import optim
from functools import partial
import copy
import math
import random
from randaugment import RandAugmentMC
from PIL import Image

from core import (
    normalise, 
    transpose, 
    pad, 
    preprocess, 
    PiecewiseLinear, 
    map_nested,
    Timer,
    group_by_key,
    Table,
    union,
    Crop,
    FlipLR,
    flip_lr
)

from torch_backend import (
    cifar10, 
    cifar10_mean, 
    cifar10_std, 
    cifar10_classes, 
    cov, 
    patches, 
    eigens,
    to,
    trainable_params,
    Flatten,
    Mul,
    GhostBatchNorm,
    GPUBatches,
)

from dawn_utils import tsv


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--mu', type=int, default=1)
parser.add_argument('--file_name', type=str, default="log.tsv")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--threshold', type=float, default=0.95)
parser.add_argument('--ema_step', type=int, default=5)
parser.add_argument('--ema_momentum', type=float, default=0.99)
parser.add_argument('--ulambda', type=float, default=1)


STEP = 0


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class ConvBN(nn.Module):
    def __init__(self, c_in, c_out, pool=None):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = pool
        self.bn = GhostBatchNorm(c_out, num_splits=16, weight_freeze=True)
        self.relu = nn.CELU(alpha=0.3)

    def forward(self, x):
        out = self.conv(x)
        if self.pool is not None:
            out = self.pool(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class WhiteningFilter(nn.Module):
    def __init__(self, Λ, V, eps=1e-2):
        super().__init__()

        filt = nn.Conv2d(3, 27, kernel_size=(3,3), padding=(1,1), bias=False)
        filt.weight.data = (V/torch.sqrt(Λ+eps)[:,None,None,None])
        filt.weight.requires_grad = False 
        
        self.filt = filt

    def forward(self, x):
        return self.filt(x)

class WhiteningBlock(nn.Module):
    def __init__(self, c_in, c_out,  Λ=None, V=None, eps=1e-2):
        super().__init__()

        self.whitening = WhiteningFilter(Λ, V, eps)
        self.layers = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(27, c_out, kernel_size=(1, 1), bias=False)),
            ('bn', GhostBatchNorm(c_out, num_splits=16, weight_freeze=True)),
            ('relu', nn.CELU(alpha=0.3))
        ]))

    def forward(self, x):
        out = self.whitening(x)
        out = self.layers(out)
        return out


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.conv1 = ConvBN(c, c)
        self.conv2 = ConvBN(c, c)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ResNet9(nn.Module):
    def __init__(self, weight, Λ, V):
        super().__init__()

        channels = [64, 128, 256, 512]

        residuals = [False, True, False, True]

        self.layers = []

        self.layers.append(
            WhiteningBlock(3, channels[0], Λ, V)
        )

        pool = nn.MaxPool2d(2)

        for i in range(1, len(channels)):
            self.layers.append(
                ConvBN(channels[i-1], channels[i], pool=pool)
            )

            if residuals[i]:
                self.layers.append(
                    Residual(channels[i])
                )

        self.layers.extend([
            nn.MaxPool2d(4),
            Flatten(),
            nn.Linear(channels[-1], 10, bias=False),
            Mul(weight)
        ])

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out

    def half(self):
        for n, p in self.named_parameters():
            if "bn" not in n:
                p.data = p.data.half()

        return self

class LabelSmoothingLoss:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, logits, targets):
        log_probs = F.log_softmax(logits, -1, _stacklevel=5)

        cross_entropy = F.nll_loss(log_probs, targets, reduction='none')
        kl = -log_probs.mean(dim=-1)

        loss = (1 - self.alpha) * cross_entropy + self.alpha * kl

        return loss.sum()


class Transform(Dataset):
    def __init__(self, dataset, device, transforms=None, to_pil=False):
        super().__init__()
        self.data, self.targets = dataset["data"], dataset["targets"]

        if to_pil:
            func = NumpyToPIL()
            self.data = [func(d) for d in self.data]

        self.transforms = transforms
        self.device = device
        
    def __len__(self):
        return len(self.data)
           
    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transforms:
            data = self.transforms(data)

        return data, targets


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = tv.transforms.Compose([
            # NumpyToPIL(),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect')])
        self.strong = tv.transforms.Compose([
            # NumpyToPIL(),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=std),
            ToHalf()])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class ToHalf:
    def __call__(self, x):
        return x.half() 

class NumpyToPIL:
    def __call__(self, x):
        return Image.fromarray(x)


def update_ema(momentum, update_freq=1):
    rho = momentum**update_freq
    def step(step, model, ema_model):
        if (step % update_freq) != 0: return
        for v, ema_v in zip(model.state_dict().values(), ema_model.state_dict().values()):
            if not v.dtype.is_floating_point: continue #skip things like num_batches_tracked.
            ema_v *= rho
            ema_v += (1-rho)*v

    return step


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def train(args, device, model, ema_model, train_batches, unlabeled_batches, opts, lr_schedulers, loss_func):
    ema_func = update_ema(args.ema_momentum, args.ema_step)

    train_meter = {
        "loss": 0,
        "acc": 0,
        "mask": 0,
        "n": 0
    }

    model.train()
    ema_model.train()

    global STEP

    unlabeled_iter = iter(unlabeled_batches)

    # loop over the labeled data
    for batch in train_batches:

        for opt, scheduler in zip(opts, lr_schedulers):
            lr = scheduler(STEP)
            for param_group in opt.param_groups:
                param_group['lr'] = lr

        inputs_x, targets_x = batch
        # (inputs_u_w, inputs_u_s), _ = u_batch

        try:
            (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabeled_batches)
            (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

        batch_size = inputs_x.shape[0]
        inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(device)
        targets_x = targets_x.to(device)
        logits = model(inputs)
        logits = de_interleave(logits, 2*args.mu+1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits

        # Lx = loss_func(logits_x, targets_x)
        Lx = F.cross_entropy(logits_x, targets_x, reduction='none').mean()

        pseudo_label = torch.softmax(logits_u_w.detach()/1, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u,
                                reduction='none') * mask).mean()

        loss = 0.5 * (Lx + args.ulambda * Lu) * batch_size

        loss.backward()

        for opt in opts:
            opt.step()
            opt.zero_grad()

        train_meter["loss"] += loss.item()
        train_meter["acc"] += (logits_x.max(dim=-1)[1] == targets_x).sum().item()
        train_meter["n"] += batch_size
        train_meter["mask"] += mask.mean().item() * batch_size

        ema_func(STEP, model, ema_model)

        STEP += 1

    train_meter["loss"] = train_meter["loss"] / train_meter["n"]
    train_meter["acc"] = train_meter["acc"] / train_meter["n"]
    train_meter["mask"] = train_meter["mask"] / train_meter["n"]

    del train_meter["n"]

    return train_meter


def warmup_cudnn(loss_func, batch_sizes, device):
    random_batch = lambda batch_size:  {
        'input': torch.Tensor(np.random.rand(batch_size,3,32,32)).cuda().half(), 
        'target': torch.LongTensor(np.random.randint(0,10,batch_size)).cuda()
    }

    random_data = torch.tensor(np.random.randn(1000,3,32,32).astype(np.float16), device=device)
    Λ, V = eigens(patches(random_data))

    model = ResNet9(weight=1/16, Λ=Λ, V=V).to(device).half()

    for size in batch_sizes:
        
        batch = random_batch(size)
        inputs, targets = batch["input"], batch["target"]

        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction='none').sum()

        model.zero_grad()
        loss.backward()

        torch.cuda.synchronize()

@torch.no_grad()
def test(device, model, ema_model, test_batches, loss_func, tta=None):
    meter = {
        "loss": 0,
        "acc": 0,
        "n": 0
    }

    model.eval()
    ema_model.eval()

    for batch in test_batches:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        # inputs, targets = batch["input"], batch["target"]

        if tta:
            logits = torch.mean(torch.stack([ema_model(t(inputs)) for t in tta], dim=0), dim=0)
        else:
            logits = ema_model(inputs)
        # loss = loss_func(logits, targets)

        loss = F.cross_entropy(logits, targets, reduction='none').sum()

        meter["loss"] += loss.item()
        meter["acc"] += (logits.max(dim=-1)[1] == targets).sum().item()
        meter["n"] += inputs.shape[0]

    meter["loss"] = meter["loss"] / meter["n"]
    meter["acc"] = meter["acc"] / meter["n"]

    del meter["n"]

    return meter


def x_u_split(labels, expand_samples, num_labeled, num_classes):
    # https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/cifar.py
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == num_labeled

    num_expand_x = math.ceil(expand_samples / num_labeled)
    labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


if __name__ == "__main__":
    device = "cuda"

    args = parser.parse_args()
    set_seed(args.seed)

    print(args)
    
    print('Downloading datasets')
    # dataset = map_nested(torch.tensor, cifar10(args.data_dir))

    dataset = cifar10(args.data_dir)

    epochs, ema_epochs = args.epochs, 2
    lr_schedule = PiecewiseLinear([0, epochs/5, epochs-ema_epochs], [0, 1.0, 0.1])
    batch_size = args.batch_size

    # train_transforms = [Crop(32, 32), FlipLR()]

    loss_func = LabelSmoothingLoss(0.2)

    print('Warming up torch')
    warmup_cudnn(
        loss_func, 
        [batch_size, batch_size*(2*args.mu+1), len(dataset['valid']['targets']) % batch_size], # normal batch size and val last batch size
        device
    )

    print('Starting timer')
    timer = Timer(synch=torch.cuda.synchronize)

    # Separate dataset
    labeled_idx, unlabeled_idx = x_u_split(
        dataset['train']['targets'], 
        50000, 
        40, 
        10
    )

    print(len(labeled_idx), len(unlabeled_idx))

    print('Preprocessing training data')
    # dataset = map_nested(to(device), dataset)
    # T = lambda x: torch.tensor(x, dtype=torch.float16, device=device)
    T = lambda x: torch.tensor(x, dtype=torch.float16)
    transforms = [
        # torch.tensor
        # to(dtype=torch.float16),
        T,
        partial(normalise, mean=T(cifar10_mean), std=T(cifar10_std)),
        partial(transpose, source='NHWC', target='NCHW'), 
    ]
    train_set = preprocess(dataset['train'], transforms)

    Λ, V = eigens(patches(train_set['data'][:10000,:])) #center crop to remove padding
    model = ResNet9(weight=1/16, Λ=Λ, V=V).to(device).half()  

    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    test_set = preprocess(dataset['valid'], transforms)

    print(f'Finished in {timer():.2} seconds')

    tensor_mean, tensor_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)

    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(p=0.5),
        tv.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    ])

    unlabeled_transforms = TransformFixMatch(tensor_mean, tensor_std)

    train_set = Transform(train_set, device, train_transforms)
    train_set = Subset(train_set, labeled_idx)

    unlabeled_set = Transform(dataset['train'], device, unlabeled_transforms, to_pil=True)
    unlabeled_set = Subset(unlabeled_set, unlabeled_idx)
    

    unlabeled_workers = 4 if args.mu == 1 else 8
    train_batches = DataLoader(train_set, batch_size, num_workers=4, shuffle=True, drop_last=True)
    unlabeled_batches = DataLoader(unlabeled_set, batch_size*args.mu, num_workers=unlabeled_workers, shuffle=True, drop_last=True)
    test_batches = DataLoader(Transform(test_set, device, None), batch_size, num_workers=4, shuffle=False, drop_last=False)

    # train_batches = GPUBatches(batch_size=batch_size, transforms=train_transforms, dataset=train_set, shuffle=True,  drop_last=True, max_options=200)
    # test_batches = GPUBatches(batch_size=batch_size, dataset=test_set, shuffle=False, drop_last=False)

    is_bias = group_by_key(('bias' in k, v) for k, v in trainable_params(model).items())

    schedules = [
        lambda step: lr_schedule((step+1)/len(train_batches))/batch_size,
        lambda step: lr_schedule((step+1)/len(train_batches))*(64/batch_size)
    ]
    
    opts = [
        optim.SGD(is_bias[False], lr=schedules[0](0), weight_decay=5e-4*batch_size, momentum=0.9, nesterov=True),
        optim.SGD(is_bias[True], lr=schedules[1](0), weight_decay=5e-4*batch_size/64, momentum=0.9, nesterov=True)
    ]

    logs = Table()

    ema_model = copy.deepcopy(model)

    for epoch in range(1, epochs+1):
        train_summary = train(args, device, model, ema_model, train_batches, unlabeled_batches, opts, schedules, loss_func)
        train_time = timer()
        test_summary = test(device, model, ema_model, test_batches, loss_func, [lambda x: x, flip_lr])
        test_time = timer(include_in_total=False)

        log = {
            "train": union({"time": train_time}, train_summary), 
            "valid": union({"time": test_time}, test_summary), 
            "total time": timer.total_time
        }

        logs.append(union({"epoch": epoch}, log))

    with open(os.path.join(os.path.expanduser(args.log_dir), args.file_name), 'w') as f:
        f.write(tsv(logs.log))
