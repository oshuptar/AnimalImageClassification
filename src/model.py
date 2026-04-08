import torch.nn as nn

class ConvolutionModel(nn.Module):
    def __init__(self, num_classes = 2, in_channels = 3, out_channels = 3):
        super().__init__()
        flattened_length: int = out_channels * 128 * 128
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_length, out_features=num_classes)
        )

    def forward(self, x):
        return self.net(x)
    