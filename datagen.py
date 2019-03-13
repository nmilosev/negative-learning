from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import time
from net import Net
import pickle

def mnist_loader(train=False):
    return torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=train, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=192, shuffle=False)


def save(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as f:
         return pickle.load(f)

def generate():
    train_loader = mnist_loader(train=True)
    test_loader = mnist_loader()
    test_loader_vertical_cut = mnist_loader()
    test_loader_horizontal_cut = mnist_loader()
    test_loader_diagonal_cut = mnist_loader()
    test_loader_triple_cut = mnist_loader()
    
    print('Generating new test sets...')

    for num in tqdm(range(0, 10000)):
        for x in range(28):
            for y in range(28):
                if y < 14:
                    test_loader_vertical_cut.dataset.test_data[num, x, y] = 0
                if x < 14:
                    test_loader_horizontal_cut.dataset.test_data[num, x, y] = 0
                if (x < 14 and y > 14) or (x > 14 and y < 14):
                    test_loader_diagonal_cut.dataset.test_data[num, x, y] = 0
                if (5 < x < 15 and 5 <  y < 15) or (17 < x < 27 and 10 < y < 20) or (7 < x < 17 and 16 < y < 26):
                    test_loader_triple_cut.dataset.test_data[num, x, y] = 0

    save('data/train_loader.pickle', train_loader)
    save('data/test_loader.pickle', test_loader)
    save('data/test_loader_vcut.pickle', test_loader_vertical_cut)
    save('data/test_loader_hcut.pickle', test_loader_horizontal_cut)
    save('data/test_loader_dcut.pickle', test_loader_diagonal_cut)
    save('data/test_loader_tcut.pickle', test_loader_triple_cut)

    print('Datasets saved')
