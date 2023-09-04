import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        efficient_net = EfficientNet.from_pretrained('efficientnet-b4')
        efficient_net._fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=efficient_net._fc.in_features, out_features=18)
        )
        self.efficient_model = efficient_net
        self.sig = nn.Sigmoid()

        weights = torch.load('/content/efficientnet-b4-6ed6700e.pth')
        self.load_state_dict(state_dict=weights, strict=False)

    def forward(self, x):
        out = self.sig(self.efficient_model(x))
        return out
