import torch.nn as nn

class ConvolutionModel(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 32, num_classes = 2):
        super().__init__()
        flattened_length: int = out_channels * 64 * 64
        self.net = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Linear(in_features=flattened_length, out_features=num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        )
    
    def forward(self, x):
        return self.net(x)
    