import torch
import torch.nn as nn
import torch.nn.functional as F

from types import MethodType
import pdb
import math


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def meps_wrapper(model, unique_labeled_trainloader, args):
    """
    1) Defines new forward method that uses MEPS pairwise similarity to compute logits
    2) Sets the model.forward method to the new MEPS forward method
    """

    n_classes = args.n_classes
    unique_labeled_trainiter = iter(unique_labeled_trainloader)
    # (labeled_x, aug_labeled_x), labeled_y = next(iter()) 
    # labeled_x.to(device), aug_labeled_x.to(device), labeled_y.to(device)
    # _, counts = torch.unique(train_y, return_counts=True)
    # assert torch.all(counts//counts.max() == 1), 'Must have an equal number of examples per class'
    # n_classes = torch.max(train_y).item() + 1
    # reorder_indices = torch.argsort(train_y)
    # train_y = train_y[reorder_indices]
    # train_x = train_x[reorder_indices]
    # aug_train_x = aug_train_x[reorder_indices]

    # N = train_x.shape[0]
    # num_splits = 16

    # padded_train_x = train_x
    # padded_aug_train_x = aug_train_x
    # if N % num_splits != 0:
    #     new_N = math.ceil(N / num_splits) * num_splits
    #     k = new_N - N
    #     perm = torch.randperm(N)
    #     idx = perm[:k]
    #     samples = train_x[idx]

    #     padded_train_x = torch.cat((train_x, samples), dim=0)

    #     samples = aug_train_x[idx]
    #     padded_aug_train_x = torch.cat((aug_train_x, samples), dim=0)

    if args.meps_type == 'dwac':
        assert args.similarity in ['rbf', 'polynomial'], 'Similarity function must be a positive-definite kernel when using DWAC'

    if args.similarity == 'rbf':
        _similarity_func = rbf_similarity
    elif args.similarity == 'l2':
        _similarity_func = l2_similarity
    elif args.similarity == 'normalized_l2':
        _similarity_func = normalized_l2_similarity
    elif args.similarity == 'dot_product':
        _similarity_func = dot_product_similarity
    elif args.similarity == 'polynomial':
        _similarity_func = polynomial_similarity
    elif args.similarity == "normalized_dot_product":
        _similarity_func = normalized_dot_product_similarity

    similarity_func = lambda x1, x2: _similarity_func(x1, x2) / args.temperature

    def prepare_labeled_data(training):

        try:
            (labeled_x, aug_labeled_x), labeled_y = next(unique_labeled_trainiter)
        except:
            unique_labeled_trainiter = iter(unique_labeled_trainloader)
            (labeled_x, aug_labeled_x), labeled_y = next(unique_labeled_trainiter)

        labeled_x = labeled_x.to(args.device)
        aug_labeled_x = aug_labeled_x.to(args.device)
        labeled_y = labeled_y.to(args.device)

        prototype_x = aug_labeled_x if training else labeled_x

        _, counts = torch.unique(labeled_y, return_counts=True)
        assert torch.all(counts//counts.max() == 1), 'Must have an equal number of examples per class'
        n_classes = torch.max(labeled_y).item() + 1
        reorder_indices = torch.argsort(labeled_y)
        labeled_y = labeled_y[reorder_indices]
        prototype_x = prototype_x[reorder_indices]

        N = prototype_x.shape[0]
        num_splits = 16

        padded_train_x = prototype_x
        if N % num_splits != 0:
            new_N = math.ceil(N / num_splits) * num_splits
            k = new_N - N
            perm = torch.randperm(N)
            idx = perm[:k]
            samples = prototype_x[idx]

            padded_train_x = torch.cat((prototype_x, samples), dim=0)

        return prototype_x, padded_train_x
        

    def dwac_forward(self, x):
        x_rep = self.compute_representation(x).view(x.shape[0], -1)
        prototype_reps = self.compute_representation(train_x).view(train_x.shape[0], -1)
        sim = similarity_func(prototype_reps, x_rep)
        sim_per_class = sim.view(n_classes, sim.shape[0]//n_classes, sim.shape[1])
        total_sim = sim_per_class.sum(dim=1).transpose(0,1)
        probs = total_sim/total_sim.sum(dim=1, keepdim=True)
        logits = torch.log(probs)
        return logits

    def meps_forward(self, x):
        @torch.no_grad()
        def compute_unlabeled_similarities(x_rep, logits):
            x_rep = de_interleave(x_rep, 2*self.args.mu+1)
            logits = de_interleave(logits, 2*self.args.mu+1)

            logits_l = torch.zeros_like(
                logits[:self.args.batch_size], device=self.args.device
            )
            x_rep_u = x_rep[self.args.batch_size:].chunk(2)[0] # only take weakly augmented samples
            logits_u = logits[self.args.batch_size:].chunk(2)[0] # only take weakly augmented samples

            logits_u_s = torch.zeros_like(
                logits[self.args.batch_size:].chunk(2)[1], device=self.args.device
            )
            
            sim = similarity_func(x_rep_u, x_rep_u) # (B_u, B_u)

            # logits_u_unsup = torch.matmul(sim, logits_u) # (B_u, num_classes)

            logits_u_unsup = sim[:,:,None] * logits_u[None,:,:]

            mask = 1 - torch.eye(logits_u_unsup.shape[0]).to(self.args.device)
            
            num = torch.sum(mask[:,:,None] * logits_u_unsup, 1)

            den = mask.sum(1)[:,None]
            logits_u_unsup_scaled = num / den

            logits_final = interleave(
                torch.cat((logits_l, logits_u_unsup_scaled, logits_u_s)), 2*self.args.mu+1).to(self.args.device)

            return logits_final

        x_rep = self.compute_representation(x).view(x.shape[0], -1)

        prototype_data, padded_prototype_data = prepare_labeled_data(self.training)
        N = prototype_data.shape[0]
        prototype_reps = self.compute_representation(padded_prototype_data).view(padded_prototype_data.shape[0], -1)
        prototype_reps = prototype_reps[:N]
        sim = similarity_func(prototype_reps, x_rep)
        sim_per_class = sim.view(n_classes, sim.shape[0]//n_classes, sim.shape[1])
        logits = sim_per_class.mean(dim=1).transpose(0,1)

        if args.unsup and self.training:
            logits_u = compute_unlabeled_similarities(x_rep, logits)

            logits = logits + logits_u

        return logits

    setattr(model, 'compute_representation', model.forward)
    if args.meps_type == 'dwac':
        setattr(model, 'forward', MethodType(dwac_forward, model))
    elif args.meps_type == 'meps':
        setattr(model, 'forward', MethodType(meps_forward, model))
    
    setattr(model, "args", args)

    return model

def l2_similarity(x1, x2):
    """
    x1 : (N1 x M)
    x2 : (N2 x M)
    Returns : (N1 x N2)
    """
    # return -torch.cdist(x1, x2)
    return -torch.sum((x1[:,None,:] - x2[None,:,:]) ** 2, dim=2)

def normalized_l2_similarity(x1, x2):
    """
    x1 : (N1 x M)
    x2 : (N2 x M)
    Returns : (N1 x N2)
    """
    return l2_similarity(x1/x1.norm(dim=1, keepdim=True), x2/x2.norm(dim=1, keepdim=True))

def rbf_similarity(x1, x2):
    """
    x1 : (N1 x M)
    x2 : (N2 x M)
    Returns : (N1 x N2)
    """
    return torch.exp(l2_similarity(x1, x2))

def normalized_dot_product_similarity(x1, x2):
    x1_norm = x1/x1.norm(dim=1, keepdim=True)
    x2_norm = x2/x2.norm(dim=1, keepdim=True)
    norm_sim = torch.matmul(x1_norm, x2_norm.T)
    return norm_sim * 4

def dot_product_similarity(x1, x2):
    return x1 @ x2.T

def polynomial_similarity(x1, x2):
    return (dot_product_similarity(x1, x2) + 1)**2