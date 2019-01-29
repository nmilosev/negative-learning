from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import time

torch.manual_seed(1)

LR = 0.1
MOM = 0.5
HIDDEN = 50

class Net(nn.Module):
    def __init__(self, net_type):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, HIDDEN)

        self.fc1normal = nn.Linear(320, HIDDEN)
        self.fc1negative = nn.Linear(320, HIDDEN)

        self.fc2 = nn.Linear(HIDDEN, 10)
        self.net_type = net_type

        self.synergy = 'inactive';

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        if self.net_type == 'hybrid':
            x = torch.ones_like(x).add(x.neg())

        if self.synergy == 'synergy':
            x1 = F.relu(self.fc1normal(x))
            x2 = F.relu(self.fc1negative(x))

            x = x1 + x2
        else:
            if self.synergy == 'inactive':
                x = F.relu(self.fc1(x))
            elif self.synergy == 'normal':
                x = F.relu(self.fc1normal(x))
            elif self.synergy == 'negative':
                x = F.relu(self.fc1negative(x))

        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


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
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                model.net_type, epoch, batch_idx * len(data), len(train_loader.dataset),
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
    print('[{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        model.net_type, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}


def mnist_loader(train=False):
    return torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=train, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64 if train else 1000, shuffle=True, **kwargs)


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
            if (5 < x < 15 and 5 <  y < 15) or (17 < x < 27 and 10 < y < 20) or (7 <  x < 17 and 16 < y < 26):
                test_loader_triple_cut.dataset.test_data[num, x, y] = 0

model_synergy = Net('normal').to(device)

optimizer_synergy = optim.SGD(filter(lambda p: p.requires_grad, model_synergy.parameters()), lr=LR, momentum=MOM)

start_time = time.time()

for epoch in range(1, 10 + 1):
    train(model_synergy, device, train_loader, optimizer_synergy, epoch)

# change network type
model_synergy.net_type = 'hybrid'
# reinitialize fully connected layers
model_synergy.fc1 = nn.Linear(320, HIDDEN).cuda()
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
# reset last layer
model_synergy.fc2 = nn.Linear(HIDDEN, 10).cuda()
model_synergy.synergy = 'normal'
model_synergy.net_type = 'normal'
# reinitialize the optimizer with new params
optimizer_synergy = optim.SGD(filter(lambda p: p.requires_grad, model_synergy.parameters()), lr=LR, momentum=MOM)


for epoch in range(21, 30 + 1):
    train(model_synergy, device, train_loader, optimizer_synergy, epoch)

# Testing:

models = [model_synergy]
model_names = ['Normal:', 'HCUT:', 'VCUT:', 'DCUT:', 'TCUT:']

datasets = [test_loader, test_loader_horizontal_cut, test_loader_vertical_cut, test_loader_diagonal_cut, test_loader_triple_cut]

for i, dataset in enumerate(datasets):
    print('Testing -- ' + model_names[i])
    for model in models:
        test(model, device, dataset)

print('--- Total time: %s seconds ---' % (time.time() - start_time))

torch.save(model_synergy, 'models/model_synergy.pytorch')

print('models saved to "models"')