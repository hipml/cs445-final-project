import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self.conv_block(in_channels, 64, normalize=False),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def conv_block(self, in_channels, out_channels, stride=2, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
