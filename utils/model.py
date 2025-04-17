from torch import nn
from torch.nn import Linear

class Net(nn.Module):
   """Basic neural network architecture with two hidden layers

   Architecture: Input -> Linear -> Sigmoid -> Linear -> Sigmoid -> Linear -> Output
   """

    def __init__(self, input_dim=1, hidden_dim=900, output_dim=1):
        """Initialize network layers and activation functions

        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Number of neurons in hidden layers
            output_dim (int): Dimension of output prediction
        """
        super(Net, self).__init__()
        # Layer 1: Input to hidden layer transformation
        self.layer1 = Linear(input_dim, hidden_dim)

        # Layer 2: Non-linear activation function
        self.layer2 = nn.Sigmoid()

        # Layer 3: Hidden to hidden layer transformation
        self.layer3 = Linear(hidden_dim, hidden_dim)

        # Layer 4: Non-linear activation function
        self.layer4 = nn.Sigmoid()

        # Layer 5: Hidden to output layer transformation
        self.layer5 = Linear(hidden_dim, output_dim)


def forward(self, x):
    """Define forward pass computation"""
    # Sequential application of network layers
    x = self.layer1(x)  # Linear transformation
    x = self.layer2(x)  # Sigmoid activation
    x = self.layer3(x)  # Linear transformation
    x = self.layer4(x)  # Sigmoid activation
    x = self.layer5(x)  # Final linear transformation

    return x