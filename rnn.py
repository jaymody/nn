import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W = nn.Parameter(torch.randn([self.hidden_size, self.hidden_size]))
        self.U = nn.Parameter(torch.randn([self.input_size, self.hidden_size]))
        self.b = nn.Parameter(torch.randn([self.hidden_size]))

        self.V = nn.Parameter(torch.randn([self.hidden_size, self.output_size]))
        self.c = nn.Parameter(torch.randn([self.output_size]))

    def forward(self, X, initial_hidden_state=None):
        # x = (batch_size, time_steps, n_features)
        # initial_hidden_state = (batch_size, hidden_size)
        batch_size = X.shape[0]
        n_time_steps = X.shape[1]

        if initial_hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_size)
        else:
            h = initial_hidden_state

        y = torch.zeros(batch_size, n_time_steps, self.output_size)
        for i in range(n_time_steps):
            x = X[:, i, :]

            a = h @ self.W + x @ self.U + self.b
            h = torch.tanh(a)

            o = h @ self.V + self.c
            y[:, i, :] = F.softmax(o, dim=-1)

        return y


def train_fn(model, train_loader, criterion, optimizer, device):
    from tqdm import tqdm

    model.train()
    train_loss = 0
    n_total = 0
    for batch in tqdm(train_loader):
        # data
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        batch_size = X.shape[0]

        # feed forward
        outputs = model(X)
        output = outputs[:, -1, :]  # get output of last time step

        # compute loss
        loss = criterion(output, y)
        train_loss += loss.item() * batch_size

        # backprop and update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n_total += batch_size
    train_loss /= n_total
    return train_loss


def eval_fn(model, val_loader, criterion, device):
    from tqdm import tqdm

    model.eval()
    val_loss, n_correct, n_total = 0, 0, 0
    for batch in tqdm(val_loader):
        # data
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        batch_size = X.shape[0]

        # feed forward
        with torch.no_grad():
            outputs = model(X)
        output = outputs[:, -1, :]  # get output of last time step

        # compute loss
        loss = criterion(output, y)
        val_loss += loss.item()

        # compute predictions
        pred = torch.argmax(output, -1)
        n_correct += torch.sum(pred == y).item()
        n_total += batch_size
    val_acc = n_correct / n_total
    val_loss /= len(val_loader)
    return val_loss, val_acc


def main():
    import json
    import os

    from torch.utils.data import DataLoader

    from datasets import MnistStrokeSequencesClassificationDataset
    from utils import get_device, set_seed, train_test_split

    # hyperparams
    datafile_path = "data/mnist_stroke_sequences.json"
    train_test_ratio = 0.8
    batch_size = 64
    num_workers = 0
    n_epochs = 10
    seed = 123

    # set seed
    set_seed(seed)

    # get device
    device = get_device()

    # get data
    if not os.path.exists(datafile_path):
        MnistStrokeSequencesClassificationDataset.download_and_process(datafile_path)
    with open(datafile_path) as fi:
        data = json.load(fi)

    # train/validation split
    train_data, val_data = train_test_split(data, train_test_ratio, shuffle=True)

    # datasets
    train_dataset = MnistStrokeSequencesClassificationDataset(train_data)
    val_dataset = MnistStrokeSequencesClassificationDataset(val_data)

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MnistStrokeSequencesClassificationDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MnistStrokeSequencesClassificationDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # model
    class ExtractLSTMOutput(nn.Module):
        def forward(self, x):
            out, _ = x
            return out

    model = nn.Sequential(
        nn.LSTM(3, 64, num_layers=1, batch_first=True),
        ExtractLSTMOutput(),
        nn.LSTM(64, 10, num_layers=1, batch_first=True),
        ExtractLSTMOutput(),
    )
    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # train loop
    for epoch in range(n_epochs):
        # train
        train_loss = train_fn(model, train_loader, criterion, optimizer, device)

        # evaluate
        val_loss, val_acc = eval_fn(model, val_loader, criterion, device)

        # log results
        print(
            f"Epoch {epoch+1}/{n_epochs}\t"
            f"loss {train_loss:.5f}\t"
            f"val_loss {val_loss:.5f}\t"
            f"val_acc {val_acc:.5f}\t"
        )


if __name__ == "__main__":
    main()
