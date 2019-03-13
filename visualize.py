import matplotlib.pyplot as plt
import sys
import torch
from net import Net

nn = torch.load(sys.argv[1], map_location='cpu')

print(nn)

convs = nn.conv2.weight.data.numpy()

i = 0
for conv_group in convs:
    for conv in conv_group:
        plt.imsave(f'convs/{i}.png', conv, cmap='gray')
        i += 1

print('done')
