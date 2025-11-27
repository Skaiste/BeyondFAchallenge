import torch
import torch.nn as nn

class BFANet(nn.Module):
    def __init__(self, task, input_dim=128, hidden_dim=32):
        super().__init__()
        self.task = task

        if task == 'cognitive_status':
            output_dim = 3
        else:
            output_dim = 1

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, feature):
        output = self.network(feature)
        if self.task == 'sex':
            output = torch.sigmoid(output)
        
        return output
