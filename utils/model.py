from torch import nn
from torch.nn import Linear

class Net(nn.Module):
    """基础网络结构"""

    def __init__(self, input_dim=1, hidden_dim=900, output_dim=1):
        super(Net, self).__init__()
        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = nn.Sigmoid()
        self.layer3 = Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Sigmoid()
        self.layer5 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x
