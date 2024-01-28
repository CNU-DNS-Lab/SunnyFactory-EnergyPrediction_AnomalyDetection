import os
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_sequential(data, seq_len=24, normalize=True):
    total_len = seq_len * 2

    seqeunces = [
        data[i:(i + total_len)]
        for i in range(len(data) - total_len + 1)
    ]

    X = np.array([seqeunce[:seq_len] for seqeunce in seqeunces], dtype='float32')
    Y = np.array([seqeunce[seq_len:] for seqeunce in seqeunces], dtype='float32')

    if normalize:
        max_val = np.max(data)
        X = X / max_val
        Y = Y / max_val
        return X, Y, max_val
    else:
        return X, Y


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(self.Y[index])


def train(model, optimizer, criterion, epochs, train_loader, val_loader, device, path):
    train_loss = []
    val_loss = []
    start = time.time()
    best_epoch = 0
    best_loss = np.inf
    model.to(device)

    for epoch in range(epochs):
        start_ = time.time()

        # train
        model.train()
        avg_loss = 0
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            X, y = data
            X, y = X.to(device), y.to(device)

            preds = model(X)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            for data in tqdm(val_loader):
                X, y = data
                X, y = X.to(device), y.to(device)

                preds = model(X)
                loss = criterion(preds, y)

                avg_val_loss += loss.item() / len(val_loader)

        end_ = time.time()

        train_loss.append(avg_loss)
        val_loss.append(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{epochs}] {end_ - start_:.2f}s')
        print(f'loss: {avg_loss:.4f} | val_loss: {avg_val_loss:.4f}')

        if avg_val_loss <= best_loss:
            best_epoch = epoch + 1
            best_loss = avg_val_loss
            torch.save(model.cpu().state_dict(), path + f'_{best_epoch}.pth')
            model.to(device)

    end = time.time()
    print(f'Train time: {end - start:.2f}s')
    print(f'Best Epoch: {best_epoch}, Best mse: {best_loss:.4f}')
    return train_loss, val_loss


def train_with_reverse_schedule_sampling(model, optimizer, criterion, epochs, train_loader, val_loader, device, path, ratio):
    train_loss = []
    val_loss = []
    start = time.time()
    best_epoch = 0
    best_loss = np.inf
    model.to(device)

    for epoch in range(epochs):
        start_ = time.time()

        # train
        model.train()
        avg_loss = 0
        ratio *= .8
        ratio = 0 if ratio < .1 else ratio
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            X, y = data
            X, y = X.to(device), y.to(device)

            preds = model(X, ratio)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            for data in tqdm(val_loader):
                X, y = data
                X, y = X.to(device), y.to(device)

                preds = model(X)
                loss = criterion(preds, y)

                avg_val_loss += loss.item() / len(val_loader)

        end_ = time.time()

        train_loss.append(avg_loss)
        val_loss.append(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{epochs}] {end_ - start_:.2f}s')
        print(f'loss: {avg_loss:.4f} | val_loss: {avg_val_loss:.4f}')

        if avg_val_loss <= best_loss:
            best_epoch = epoch + 1
            best_loss = avg_val_loss
            torch.save(model.cpu().state_dict(), path + f'_{best_epoch}.pth')
            model.to(device)

    end = time.time()
    print(f'Train time: {end - start:.2f}s')
    print(f'Best Epoch: {best_epoch}, Best mse: {best_loss:.4f}')
    return train_loss, val_loss
