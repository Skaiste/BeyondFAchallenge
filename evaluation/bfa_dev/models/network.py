import torch
import torch.nn as nn
from collections import OrderedDict

class BFANet(nn.Module):
    def __init__(self, task, input_dim=128, hidden_dim=32, reduce_hidden_dim=True):
        super().__init__()
        self.task = task

        output_dim = 3 if task == 'cognitive_status' else 1
        hidden_output_dim = hidden_dim // 4 if reduce_hidden_dim else hidden_dim

        layers = OrderedDict([
            ('linear1', nn.Linear(input_dim, hidden_dim)),
            ('relu1', nn.LeakyReLU(0.2)),
            ('linear2', nn.Linear(hidden_dim, hidden_dim)),
            ('relu2', nn.LeakyReLU(0.2)),
        ])
        if reduce_hidden_dim:
            layers.update([
                ('linear3', nn.Linear(hidden_dim, hidden_dim //2)),
                ('relu3', nn.LeakyReLU(0.2)),
                ('linear3', nn.Linear(hidden_dim //2, hidden_dim //4)),
                ('relu3', nn.LeakyReLU(0.2)),
            ])
        
        layers.update([
            ('final_linear', nn.Linear(hidden_output_dim, output_dim))
        ])
        self.network = nn.Sequential(layers)

    def forward(self, feature):
        output = self.network(feature)
        if self.task == 'sex':
            output = torch.sigmoid(output)
        
        return output
