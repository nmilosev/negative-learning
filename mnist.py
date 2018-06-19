from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import sys


torch.manual_seed(42)


class Net(nn.Module):
    def __init__(self, net_type):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(288, 512)
        self.fc2 = nn.Linear(512, 10)
        self.net_type = net_type

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(F.max_pool2d(self.conv2(x), 2))
        x = F.leaky_relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 288)
        #print(x.size())
        if self.net_type == 'negative':  # -x
            x = x.neg()
        if self.net_type == 'negative_relu' or self.net_type == 'hybrid':  # 1 - x
            x = torch.ones_like(x).add(x.neg())
        x = F.leaky_relu(self.fc1(x))
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
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('[{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        model.net_type, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def mnist_loader(train=False):
    return torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=train, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64 if train else 1000, shuffle=True, **kwargs)


train_loader = mnist_loader(train=True)
test_loader = mnist_loader()
test_loader_vertical_cut1 = mnist_loader()
test_loader_vertical_cut2 = mnist_loader()
test_loader_horizontal_cut1 = mnist_loader()
test_loader_horizontal_cut2 = mnist_loader()
test_loader_diagonal_cut1 = mnist_loader()
test_loader_diagonal_cut2 = mnist_loader()

print('Generating additional datasets...')

if '--skipgen' not in sys.argv:
    for num in range(0, 10000):
        for x in range(28):
            for y in range(28):
                if y < 14:
                    test_loader_vertical_cut1.dataset.test_data[num, x, y] = 0
                if y >= 14:
                    test_loader_vertical_cut2.dataset.test_data[num, x, y] = 0
                if x < 14:
                    test_loader_horizontal_cut1.dataset.test_data[num, x, y] = 0
                if x >= 14:
                    test_loader_horizontal_cut2.dataset.test_data[num, x, y] = 0
                if (x < 14 and y > 14) or (x > 14 and y < 14):
                    test_loader_diagonal_cut1.dataset.test_data[num, x, y] = 0
                if (x >= 14 and y < 14) or (x < 14 and y >= 14):
                    test_loader_diagonal_cut2.dataset.test_data[num, x, y] = 0


model_normal = Net('normal').to(device)
model_negative = Net('negative').to(device)
model_negative_relu = Net('negative_relu').to(device)
model_hybrid = Net('normal').to(device)

optimizer_normal = optim.SGD(filter(lambda p: p.requires_grad, model_normal.parameters()), lr=0.05, momentum=0.5)
optimizer_negative = optim.SGD(filter(lambda p: p.requires_grad, model_negative.parameters()), lr=0.05, momentum=0.5)
optimizer_negative_relu = optim.SGD(filter(lambda p: p.requires_grad, model_negative_relu.parameters()), lr=0.05, momentum=0.5)
optimizer_hybrid = optim.SGD(filter(lambda p: p.requires_grad, model_hybrid.parameters()), lr=0.05, momentum=0.5)

start_time = time.time()

for epoch in range(1, 10 + 1):
    train(model_normal, device, train_loader, optimizer_normal, epoch)

for epoch in range(1, 10 + 1):
    train(model_negative, device, train_loader, optimizer_negative, epoch)

for epoch in range(1, 10 + 1):
    train(model_negative_relu, device, train_loader, optimizer_negative_relu, epoch)

for epoch in range(1, 10 + 1):
    train(model_hybrid, device, train_loader, optimizer_hybrid, epoch)

# change network type
model_hybrid.net_type = 'hybrid'
# reinitialize fully connected layers
model_hybrid.fc1 = nn.Linear(288, 512).cuda()
model_hybrid.fc2 = nn.Linear(512, 10).cuda()
# freeze convolutional layers
model_hybrid.conv1.weight.requires_grad = False
model_hybrid.conv2.weight.requires_grad = False
model_hybrid.conv3.weight.requires_grad = False
# reinitialize the optimizer with new params
optimizer_hybrid = optim.SGD(filter(lambda p: p.requires_grad, model_hybrid.parameters()), lr=0.05, momentum=0.5)

for epoch in range(11, 15 + 1):
    train(model_hybrid, device, train_loader, optimizer_hybrid, epoch)

loaders = [test_loader, test_loader_horizontal_cut1, test_loader_horizontal_cut2,
            test_loader_vertical_cut1, test_loader_vertical_cut2, test_loader_diagonal_cut1,
            test_loader_diagonal_cut2]
loader_names = ['Normal data set', 'Horizontal cut LEFT', 'Horizontal cut RIGHT', 'Vertical cut TOP',
                'Vertical cut BOTTOM', 'Diagonal cut RTL', 'Diagonal cut LTR']

for loader, name in zip(loaders, loader_names):
    print(name)
    test(model_normal, device, loader)
    test(model_negative, device, loader)
    test(model_negative_relu, device, loader)
    test(model_hybrid, device, loader)

print('--- Total time: %s seconds ---' % (time.time() - start_time))

torch.save(model_normal, 'models/model_normal.pytorch')
torch.save(model_negative, 'models/model_negative.pytorch')
torch.save(model_negative_relu, 'models/model_negative_relu.pytorch')
torch.save(model_hybrid, 'models/model_hybrid.pytorch')

print('models saved to "models"')
