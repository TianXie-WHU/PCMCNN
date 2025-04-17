import torch
from utils.data_loader import read_xlrd
import torch.utils.data as Data
from torch.utils.data import random_split
from utils.metrics import cal_Rsqure, cal_MAE
from utils.model import Net
from torch import nn

if __name__ == '__main__':
    # Data preparation and preprocessing
    excelFile1 = 'data/TGLAND.xlsx'
    # Load input features from 'water' sheet
    x = read_xlrd(excelFile=excelFile1, sheet_name='water')
    # Load target values from 'e1' sheet
    y = read_xlrd(excelFile=excelFile1, sheet_name='e1')

    # Convert numpy arrays to PyTorch tensors
    x1 = torch.FloatTensor(x)
    y1 = torch.FloatTensor(y)

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    # Create dataset and split into train/test sets (80/20 split)
    dataset = Data.TensorDataset(x, y)
    train_data, test_data=random_split(dataset, [round(0.8*x.shape[0]),round(0.2*x.shape[0])])
    # Create data loaders with full-batch training (batch_size=10000)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10000, shuffle=True, num_workers=0,)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=True, num_workers=0,)

    # Model initialization
    net = Net()
    # Enable GPU acceleration if available
    if torch.cuda.is_available():
        net = net.cuda()

    # Loss function configuration
    loss_fn = nn.MSELoss()# Mean Squared Error loss
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    # Optimizer configuration
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Training configuration parameters
    total_train_step = 0 # Counter for training iterations
    total_test_step = 0 # Counter for testing iterations
    epoch = 10000 # Maximum training epochs
    # Main training loop
    for i in range(epoch):
        net.train() # Set model to training mode
        for data in train_loader:
            x, y = data
            # Move data to GPU if available
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # Forward pass
            outputs = net(x)
            trainloss = loss_fn(outputs, y)

            # Backpropagation
            optimizer.zero_grad()
            trainloss.backward()
            optimizer.step()

            # Training progress monitoring
            total_train_step = total_train_step + 1
            if total_train_step % 10 == 0:
                print("Training step：{}, Loss: {}".format(total_train_step, trainloss.item()))

        # Validation phase
        net.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            for test in test_loader:
                m, n = data
                if torch.cuda.is_available():
                    m = m.cuda()
                    n = n.cuda()
                output = net(m)
                testloss = loss_fn(output, n)
                # Calculate evaluation metrics
                R2 = cal_Rsqure(n, output)
                mae = cal_MAE(n, output)
                print("Test Loss: {}".format(testloss.item()))
                print("Test R-squared: {}".format(R2))
                print("Test MAE: {}".format(mae))

        # Early stopping and model saving condition
        if(abs(testloss.item() - trainloss.item()) <= 0.01) and (R2 >= 0.95):
            torch.save(net, "net1.pth")
            print("Model saved successfully")
            break