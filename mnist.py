from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import time
from net import Net
import pickle

torch.manual_seed(1)

LR = 0.1
MOM = 0.5
HIDDEN = 50


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('[net_type: {}, synergy: {}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                model.net_type, model.synergy, epoch, batch_idx * len(data), len(train_loader.dataset),
                                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('[net_type: {}, synergy: {}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        model.net_type, model.synergy, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}

from datagen import generate, load

if '--generate' in sys.argv:
    generate()

train_loader = load('data/train_loader.pickle')
test_loader = load('data/test_loader.pickle')
test_loader_vertical_cut = load('data/test_loader_vcut.pickle')
test_loader_horizontal_cut = load('data/test_loader_hcut.pickle')
test_loader_diagonal_cut = load('data/test_loader_dcut.pickle')
test_loader_triple_cut = load('data/test_loader_tcut.pickle')

model_synergy = Net('normal').to(device)

optimizer_synergy = optim.SGD(filter(lambda p: p.requires_grad, model_synergy.parameters()), lr=LR, momentum=MOM)

start_time = time.time()

for epoch in range(1, 10 + 1):
    train(model_synergy, device, train_loader, optimizer_synergy, epoch)

# change network type
model_synergy.net_type = 'negative'
# reinitialize fully connected layers
model_synergy.fc1 = nn.Linear(30, HIDDEN).cuda()
model_synergy.fc2 = nn.Linear(HIDDEN, 10).cuda()
# freeze convolutional layers
model_synergy.conv1.weight.requires_grad = False
model_synergy.conv2.weight.requires_grad = False
# activate synergy to negative
model_synergy.synergy = 'negative'
# reinitialize the optimizer with new params
optimizer_synergy = optim.SGD(filter(lambda p: p.requires_grad, model_synergy.parameters()), lr=LR, momentum=MOM)

for epoch in range(11, 20 + 1):
    train(model_synergy, device, train_loader, optimizer_synergy, epoch)

# next 10 epochs are going to be for normal part
model_synergy.synergy = 'normal'
model_synergy.net_type = 'normal'
# reinitialize the optimizer with new params
optimizer_synergy = optim.SGD(filter(lambda p: p.requires_grad, model_synergy.parameters()), lr=LR, momentum=MOM)

for epoch in range(21, 30 + 1):
    train(model_synergy, device, train_loader, optimizer_synergy, epoch)

# Testing:

models = [model_synergy]
dataset_names = ['Normal:', 'HCUT:', 'VCUT:', 'DCUT:', 'TCUT:']

datasets = [test_loader, test_loader_horizontal_cut, test_loader_vertical_cut, test_loader_diagonal_cut, test_loader_triple_cut]

for i, dataset in enumerate(datasets):
    print('Testing (fc1normal active only) -- ' + dataset_names[i])
    test(model_synergy, device, dataset)

model_synergy.synergy = 'negative'
model_synergy.net_type = 'negative'

for i, dataset in enumerate(datasets):
    print('Testing (fc1negative active only) -- ' + dataset_names[i])
    test(model_synergy, device, dataset)

model_synergy.synergy = 'synergy'
for i, dataset in enumerate(datasets):
    print('Testing (synergy) -- ' + dataset_names[i])
    test(model_synergy, device, dataset)

print('--- Total time: %s seconds ---' % (time.time() - start_time))

torch.save(model_synergy, 'models/model_synergy.pytorch')

print('models saved to "models"')

#print('testing false positives etc.')
#
#def test_csv(model, device, test_loader, writer):
#    model.eval()
#    index = 1
#    with torch.no_grad():
#        for data, target in test_loader:
#            data, target = data.to(device), target.to(device)
#
#            correct = target.item()
#
#            model_synergy.synergy = 'normal'
#            model_synergy.net_type = 'normal'
#
#            output_normal = model(data).max(1, keepdim=True)[1].item()
#
#            model_synergy.synergy = 'negative'
#            model_synergy.net_type = 'negative'
#
#            output_negative = model(data).max(1, keepdim=True)[1].item()
#
#            model_synergy.synergy = 'synergy'
#
#            output_synergy = model(data).max(1, keepdim=True)[1].item()
#            if not (correct == output_normal == output_negative == output_synergy):
#                writer.writerow([index, correct, output_normal, output_negative, output_synergy])
#
#            index += 1
#
#import csv
#
#with open('out.csv', 'w') as csv_file:
#    writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
#    writer.writerow(['#', 'correct-label', 'normal-prediction', 'negative-prediction', 'synergy-prediction'])
#    for i, dataset in enumerate(datasets):
#        writer.writerow([f'Dataset: {dataset_names[i]}'])
#        test_csv(model_synergy, device, dataset, writer)


