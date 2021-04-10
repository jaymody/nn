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


def train_fn(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    for batch in train_loader:
        # load data to device
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)

        # feed forwards
        output = model(x)

        # compute loss
        loss = criterion(output, y)
        train_loss += loss.item() / batch_size

        # backprop and step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return train_loss


def eval_fn(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    n_correct = 0
    n_total = 0
    for batch in val_loader:
        # load data to device
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)

        # feed forward
        with torch.no_grad():
            output = model(x)

        # compute loss
        loss = criterion(output, y)
        val_loss += loss.item() / batch_size

        # get predictions
        pred = torch.argmax(output, axis=-1)

        # add number of correct and total predictions made
        n_correct += torch.sum(pred == y).item()
        n_total += batch_size

    val_acc = n_correct / n_total
    return val_loss, val_acc


if __name__ == "__main__":
    # setup
    import random
    import numpy as np
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, ToTensor, Lambda

    # hyperparams
    lr = 1e-3
    weight_decay = 1e-6
    n_epochs = 20
    num_workers = 4
    batch_size = 128
    train_test_ratio = 0.8
    seed = 1234

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed for deterministic results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # load datasets
    transform = Compose([ToTensor(), Lambda(lambda x: torch.flatten(x))])
    train_dataset = MNIST(
        root="data/mnist/train", train=True, download=True, transform=transform
    )
    test_dataset = MNIST(
        root="data/mnist/test", train=False, download=True, transform=transform
    )

    # train/validation split
    train_size = int(len(train_dataset) * train_test_ratio)
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

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
        train_loss = train_fn(model, train_loader, criterion, optimizer)

        # eval
        val_loss, val_acc = eval_fn(model, val_loader, criterion)

        # log results
        print(
            f"Epoch {epoch+1}/{n_epochs}\t"
            f"loss {train_loss:.3f}\t"
            f"val_loss {val_loss:.3f}\t"
            f"val_acc {val_acc:.3f}\t"
        )

    # test
    _, test_acc = eval_fn(model, test_loader, criterion)
    print(f"\nAccuracy on Test Set: {test_acc:.3f}")
