import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_fn(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    n_total = 0
    for batch in train_loader:
        # load data to device
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        batch_size = x.shape[0]

        # feed forwards
        output = model(x)

        # compute loss
        loss = criterion(output, y)
        running_loss += loss.item() * batch_size
        n_total += batch_size

        # backprop and step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = running_loss / n_total
    return train_loss


def eval_fn(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0
    n_correct = 0
    n_total = 0
    for batch in val_loader:
        # load data to device
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        batch_size = x.shape[0]

        # feed forward
        with torch.no_grad():
            output = model(x)

        # compute loss
        loss = criterion(output, y)
        running_loss += loss.item() * batch_size

        # get predictions
        pred = torch.argmax(output, axis=-1)

        # add number of correct and total predictions made
        n_correct += torch.sum(pred == y).item()
        n_total += batch_size

    val_loss = running_loss / n_total
    val_acc = n_correct / n_total
    return val_loss, val_acc


def main():
    # setup
    import random

    import numpy as np
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, Lambda, ToTensor

    from utils import get_device, set_seed, train_test_split_dataset

    # hyperparams
    lr = 1e-3
    weight_decay = 1e-6
    n_epochs = 20
    num_workers = 4
    batch_size = 128
    train_test_ratio = 0.8
    seed = 123

    # get device
    device = get_device()

    # set seed for deterministic results
    set_seed(seed)

    # load datasets
    transform = Compose([ToTensor(), Lambda(torch.flatten)])
    train_dataset = MNIST(
        root="data/mnist/train", train=True, download=True, transform=transform
    )
    test_dataset = MNIST(
        root="data/mnist/test", train=False, download=True, transform=transform
    )

    # train/validation split
    train_dataset, val_dataset = train_test_split_dataset(
        train_dataset, train_test_ratio
    )

    # data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # model
    model = MLP(in_dim=28 * 28, hidden_dim=64, out_dim=10)
    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # train loop
    for epoch in range(n_epochs):
        # train
        train_loss = train_fn(model, train_loader, criterion, optimizer, device)

        # eval
        val_loss, val_acc = eval_fn(model, val_loader, criterion, device)

        # log results
        print(
            f"Epoch {epoch+1}/{n_epochs}\t"
            f"loss {train_loss:.3f}\t"
            f"val_loss {val_loss:.3f}\t"
            f"val_acc {val_acc:.3f}\t"
        )

    # test
    _, test_acc = eval_fn(model, test_loader, criterion, device)
    print(f"\nAccuracy on Test Set: {test_acc:.3f}")


if __name__ == "__main__":
    main()
