import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple linear layer: input dim 10 → output dim 5
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        # Pass input through the linear layer
        out = self.fc(x)
        # Return two outputs: the layer output, and the output scaled by 2
        return out, out * 2

