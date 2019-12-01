import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, datasets
from torch.nn import functional as F

from toymodel import Toynetwork
from randaugment import policies as found_policies
from randaugment import augmentation_transforms
from wide_resnet import Wide_ResNet 


def split_data(dataset):
    pool_idx = np.arange(len(dataset))
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
        #TODO add "simple_augm" here
        y_ = model(x)
    x_augm = random_augmentation(x)
    y_augm = model(x_augm)
    kl = _kl_divergence_with_logits(y_, y_augm)
    kl = torch.mean(kl)
    return kl


def simple_augm(x):
    # fix later
    pass


def random_augmentation(x):
    # They are actually performing this beforehand and 
    # saving all samples ==> saves a lot of computation
    # but also less random in the end, no?
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
    onehot = F.one_hot(ground_truth, num_classes=10)
    correct_label_probs = torch.sum(pred*onehot, -1)
    smaller_than_threshold = torch.lt(correct_label_probs, eta)
    smaller_than_threshold.requires_grad = False
    Z = np.maximum(torch.sum(smaller_than_threshold), 1)
    masked_loss = torch.log(correct_label_probs)*smaller_than_threshold # Note: they do not seem to be using log even though they say so in the paper
    loss = torch.sum(-masked_loss)
    return loss/Z

def get_next_batch(data_iter, data_loader):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        batch = next(data_iter)
    return batch, data_iter


def update_eta(T: int, k: int, step: int) -> float:
    # linear-schedule
    return (step/T)*(1 - 1/k) + 1/k


def train():
    model = Wide_ResNet(28, 2, 0.3, 10)
    data_transform = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform)
    #valset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform)
    testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)
    #unlab_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()) 

    labeled_idx, unlabeled_idx = split_data(trainset)

    subsampler_lab = torch.utils.data.SubsetRandomSampler(labeled_idx)
    #subsampler_unlab = torch.utils.data.SubsetRandomSampler(unlabeled_idx)
    labeled_trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=32,
        sampler=subsampler_lab,
        pin_memory=True,
    )
    unlabeled_trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=320,
        sampler=subsampler_lab,
        pin_memory=True,
    )

    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.003
    )
    steps = int(4e5)
    lambd = 1
    sup_batch_iterator = iter(labeled_trainloader)
    unsup_batch_iterator = iter(unlabeled_trainloader)
    classes = 10
    for step in range(steps):
        optimizer.zero_grad()
        x_lab, sup_batch_iterator = get_next_batch(sup_batch_iterator, labeled_trainloader)
        x_unlab, unsup_batch_iterator = get_next_batch(unsup_batch_iterator, unlabeled_trainloader)

        eta = update_eta(steps, classes, step)
        supervised_loss = supervised_batch(model, x_lab, eta)
                
        unsupervised_loss = unsupervised_batch(model, x_unlab)
        total_loss = supervised_loss + lambd*unsupervised_loss
        print(total_loss)
        total_loss.backward()
        optimizer.step()
        
        if step % 1000 == 0 and step != 0:
            print('Get Accuracy here')
# finns en del saker kvar i dataloading, gor samma preprocessing 


if __name__ == "__main__":
    train()