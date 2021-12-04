import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, OrderedDict
import torchvision as tv
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
from functools import partial
import copy

from SGD import SGD

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


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_dir', type=str, default='.')

STEP = 0

torch.backends.cudnn.benchmark = True

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
    def __init__(self, dataset, device, transforms=None):
        super().__init__()
        self.data, self.targets = dataset["data"], dataset["targets"]
        self.transforms = transforms
        self.device = device
        
    def __len__(self):
        return len(self.data)
           
    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transforms:
            data = self.transforms(data)

        return data, targets


def update_ema(momentum, update_freq=1):
    rho = momentum**update_freq
    def step(step, model, ema_model):
        if (step % update_freq) != 0: return
        for v, ema_v in zip(model.state_dict().values(), ema_model.state_dict().values()):
            if not v.dtype.is_floating_point: continue #skip things like num_batches_tracked.
            ema_v *= rho
            ema_v += (1-rho)*v

        # for v, ema_v in zip(model.parameters(), ema_model.parameters()):
        #     if not v.dtype.is_floating_point: continue #skip things like num_batches_tracked.
        #     ema_v.data.mul_(rho)
        #     ema_v.data.add_(v.data, alpha=1-rho)

        # for v, ema_v in zip(model.buffers(), ema_model.buffers()):
        #     if not v.dtype.is_floating_point: continue #skip things like num_batches_tracked.
        #     # ema_v.data.mul_(rho)
        #     # ema_v.data.add_(v.data, alpha=1-rho)
        #     ema_v.data.copy_(v.data)

    return step


def zero_grad(model):
    for param in model.parameters():
        param.grad = None


def train(device, model, ema_model, train_batches, opts, lr_schedulers, loss_func):
    ema_func = update_ema(0.99, 5)

    train_meter = {
        "loss": 0,
        "acc": 0,
        "n": 0
    }

    model.train()
    ema_model.train()

    global STEP

    for batch in train_batches:

        for opt, scheduler in zip(opts, lr_schedulers):
            lr = scheduler(STEP)
            for param_group in opt.param_groups:
                param_group['lr'] = lr

        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        # inputs, targets = batch["input"], batch["target"]

        logits = model(inputs)
        loss = loss_func(logits, targets)

        loss.backward()

        for opt in opts:
            opt.step()
            # opt.zero_grad()

        zero_grad(model)

        train_meter["loss"] += loss.item()
        train_meter["acc"] += (logits.max(dim=-1)[1] == targets).sum().item()
        train_meter["n"] += inputs.shape[0]

        ema_func(STEP, model, ema_model)

        STEP += 1

    train_meter["loss"] = train_meter["loss"] / train_meter["n"]
    train_meter["acc"] = train_meter["acc"] / train_meter["n"]

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
        loss = loss_func(logits, targets)

        zero_grad(model)
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
        loss = loss_func(logits, targets)

        meter["loss"] += loss.item()
        meter["acc"] += (logits.max(dim=-1)[1] == targets).sum().item()
        meter["n"] += inputs.shape[0]

    meter["loss"] = meter["loss"] / meter["n"]
    meter["acc"] = meter["acc"] / meter["n"]

    del meter["n"]

    return meter


if __name__ == "__main__":
    device = "cuda"

    args = parser.parse_args()
    
    print('Downloading datasets')
    dataset = map_nested(torch.tensor, cifar10(args.data_dir))

    epochs, ema_epochs = 10, 2
    lr_schedule = PiecewiseLinear([0, epochs/5, epochs-ema_epochs], [0, 1.0, 0.1])
    batch_size = 512
    train_transforms = tv.transforms.Compose([
        # tv.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        tv.transforms.RandomCrop(32, padding=0, padding_mode='reflect'),
        tv.transforms.RandomHorizontalFlip(p=0.5)
    ])

    # train_transforms = [Crop(32, 32), FlipLR()]

    loss_func = LabelSmoothingLoss(0.2)

    print('Warming up torch')
    warmup_cudnn(
        loss_func, 
        [batch_size, len(dataset['valid']['targets']) % batch_size], # normal batch size and val last batch size
        device
    )

    print('Starting timer')
    timer = Timer(synch=torch.cuda.synchronize)
    
    print('Preprocessing training data')
    # dataset = map_nested(to(device), dataset)
    # T = lambda x: torch.tensor(x, dtype=torch.float16, device=device)
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    transforms = [
        partial(normalise, mean=T(cifar10_mean), std=T(cifar10_std)),
        partial(transpose, source='NHWC', target='NCHW'), 
    ]
    train_set = preprocess(dataset['train'], transforms + [partial(pad, border=4), to(dtype=torch.float16)])

    Λ, V = eigens(patches(train_set['data'][:10000,:,4:-4,4:-4])) #center crop to remove padding
    model = ResNet9(weight=1/16, Λ=Λ, V=V).to(device).half()  

    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    test_set = preprocess(dataset['valid'], transforms + [to(dtype=torch.float16)])
    print(f'Finished in {timer():.2} seconds')

    train_batches = DataLoader(Transform(train_set, device, train_transforms), batch_size, num_workers=4, shuffle=True, drop_last=True)
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
        train_summary = train(device, model, ema_model, train_batches, opts, schedules, loss_func)
        train_time = timer()
        test_summary = test(device, model, ema_model, test_batches, loss_func, [lambda x: x, flip_lr])
        test_time = timer(include_in_total=False)

        log = {
            "train": union({"time": train_time}, train_summary), 
            "test": union({"time": test_time}, test_summary), 
            "total time": timer.total_time
        }

        logs.append(union({"epoch": epoch}, log))
