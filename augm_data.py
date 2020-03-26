import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import transforms, datasets
from torch.nn import functional as F
from tqdm import tqdm

from toymodel import Toynetwork
from randaugment import policies as found_policies
from randaugment import augmentation_transforms
from wide_resnet import Wide_ResNet 


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cpu")


def split_data(pool_idx):
    np.random.seed()
    labeled_idx = np.random.permutation(pool_idx)[:4000]
    unlabeled_idx = np.setdiff1d(pool_idx, labeled_idx)
    return labeled_idx, unlabeled_idx

def supervised_batch(model, batch, eta):
    x, y = batch
    y_ = model(x)
    pred = F.softmax(y_, dim=-1)
    filtered_loss = training_signal_annealing(pred, y, eta)
    return filtered_loss

def unsupervised_batch(model, batch):
    x, _ = batch
    # As suggested by Miyato et al.
    with torch.no_grad():
        y_ = model(x)
    x_augm = random_augmentation(x).float()
    y_augm = model(x_augm)
    kl = _kl_divergence_with_logits(y_, y_augm)
    kl = torch.mean(kl)
    return kl


def random_augmentation(x):
    # Original implementation performs these augmentations beforehand 
    # and saves all samples ==> saves a lot of computation
    # but also results in less variation 
    aug_policies = found_policies.randaug_policies()
    x_augm = torch.zeros_like(x)
    for i in range(x.size()[0]):
        chosen_policy = aug_policies[np.random.choice(
            len(aug_policies))]
        aug_image = augmentation_transforms.apply_policy(
            chosen_policy, x[i,:,:,:].permute(1,2,0).numpy())
        aug_image = augmentation_transforms.cutout_numpy(aug_image)
        x_augm[i,:,:,:] = torch.tensor(aug_image).permute(2,0,1)
    return x_augm


def _kl_divergence_with_logits(p_logits, q_logits):
  p = F.softmax(p_logits)
  log_p = F.log_softmax(p_logits)
  log_q = F.log_softmax(q_logits)
  kl = torch.sum(p * (log_p - log_q), -1)
  return kl


def training_signal_annealing(pred, ground_truth, eta):
    onehot = F.one_hot(ground_truth, num_classes=10).float()
    correct_label_probs = torch.sum(pred*onehot, -1)
    smaller_than_threshold = torch.lt(correct_label_probs, eta).float()
    smaller_than_threshold.requires_grad = False
    Z = np.maximum(torch.sum(smaller_than_threshold.cpu()), 1).float()
    masked_loss = torch.log(correct_label_probs)*smaller_than_threshold 
    # Note: they do not seem to be using log even though they say so in the paper
    loss = torch.sum(-masked_loss)
    return loss/Z

def get_next_batch(data_iter, data_loader):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        batch = next(data_iter)
    batch = batch[0].to(device), batch[1].to(device)
    return batch, data_iter


def update_eta(T: int, k: int, step: int) -> float:
    # linear-schedule
    return (step/T)*(1 - 1/k) + 1/k


def calculate_accuracy(model, dataloader):
    accuracy = 0
    total = 0
    for data in dataloader:
        x, label = data
        output = model(x)
        _, pred = torch.max(output, 1)
        total += label.size(0)
        accuracy += (pred == label).sum().item()
    
    return accuracy/total


def train():
    model = Wide_ResNet(28, 2, 0.3, 10)
    model.to(device)
    data_transform = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform)
    testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)

    train_idx, val_idx = train_test_split(
        np.arange(len(trainset)), test_size=0.2, shuffle=True
    )
    labeled_idx, unlabeled_idx = split_data(train_idx)
    labeled_idx_val, unlabeled_idx_val = split_data(val_idx)

    subsampler_lab = torch.utils.data.SubsetRandomSampler(labeled_idx)
    subsampler_unlab = torch.utils.data.SubsetRandomSampler(unlabeled_idx)
    subsampler_val_lab = torch.utils.data.SubsetRandomSampler(labeled_idx_val)
    subsampler_val_unlab = torch.utils.data.SubsetRandomSampler(unlabeled_idx_val)

    labeled_trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        sampler=subsampler_lab,
    )
    unlabeled_trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=320,
        sampler=subsampler_unlab,
    )
    labeled_val_trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        sampler=subsampler_val_lab,
    )
    unlabeled_val_trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=320,
        sampler=subsampler_val_unlab,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size = 32,
        drop_last = True    
    )

    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.03,
    )

    steps = int(4e5)
    lambd = 1
    sup_batch_iterator = iter(labeled_trainloader)
    unsup_batch_iterator = iter(unlabeled_trainloader)
    sup_val_batch_iterator = iter(labeled_val_trainloader)
    unsup_val_batch_iterator = iter(unlabeled_val_trainloader)
    classes = 10
    writer = SummaryWriter(os.getcwd())
    for step in tqdm(range(steps)):
        optimizer.zero_grad()
        x_lab, sup_batch_iterator = get_next_batch(sup_batch_iterator, labeled_trainloader)
        x_unlab, unsup_batch_iterator = get_next_batch(unsup_batch_iterator, unlabeled_trainloader)

        eta = update_eta(steps, classes, step)
        supervised_loss = supervised_batch(model, x_lab, eta)
        writer.add_scalar("loss/supervised", supervised_loss.detach(), step)

        unsupervised_loss = unsupervised_batch(model, x_unlab)
        writer.add_scalar("loss/unsupervised", unsupervised_loss.detach(), step)

        total_loss = supervised_loss + lambd*unsupervised_loss

        writer.add_scalar("loss/total", total_loss.detach(), step)
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_lab_val, sup_val_batch_iterator = get_next_batch(sup_val_batch_iterator, labeled_val_trainloader)
            x_unlab_val, unsup_val_batch_iterator = get_next_batch(unsup_val_batch_iterator, unlabeled_val_trainloader)

            supervised_val_loss = supervised_batch(model, x_lab_val, eta)
            writer.add_scalar("loss/val_supervised", supervised_val_loss.detach(), step)

            unsupervised_val_loss = unsupervised_batch(model, x_unlab_val)
            writer.add_scalar("loss/val_unsupervised", unsupervised_val_loss.detach(), step)

            total_val_loss = supervised_val_loss + lambd*unsupervised_val_loss
            writer.add_scalar("loss/val_total", total_val_loss.detach(), step)    
        
            if step % 200 == 0 and step != 0:
                accuracy = calculate_accuracy(model, testloader)
                writer.add_scalar("Test Accuracy", accuracy, step)


if __name__ == "__main__":
    train()