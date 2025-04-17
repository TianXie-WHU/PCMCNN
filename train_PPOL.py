import torch
from utils.data_loader import read_xlrd
import torch.utils.data as Data
from torch.utils.data import random_split
from utils.metrics import cal_Rsqure, cal_MAE
from utils.model import Net
from torch import nn

if __name__ == '__main__':
    # Data configuration and loading
    excelFile1 = 'data/TGLAND.xlsx'
    # Load input features from 'water' sheet
    x = read_xlrd(excelFile=excelFile1, sheet_name='water')
    # Load multi-target outputs from 'result' sheet
    y = read_xlrd(excelFile=excelFile1, sheet_name='result')
    # Convert numpy arrays to PyTorch tensors
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    # Create dataset and split into 80% training, 20% testing
    dataset = Data.TensorDataset(x, y)
    # Create data loaders with full-batch training strategy
    train_data, test_data=random_split(dataset, [round(0.8*x.shape[0]),round(0.2*x.shape[0])])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10000, shuffle=True, num_workers=0,)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=True, num_workers=0,)

    # Initialize and load pre-trained models
    net1 = Net() # Base model initialization
    net2 = Net()
    net3 = Net()
    net1 = torch.load("net1.pth") # Overwrite with pre-trained weights
    net2 = torch.load("net2.pth")
    net3 = torch.load("net3.pth")

    # Loss function configuration
    loss_fn = nn.MSELoss() # Mean Squared Error loss
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    # Training configuration
    total_train_step = 0 # Training iteration counter
    total_test_step = 0 # Testing iteration counter
    epoch = 100000 # Maximum training epochs

    # Individual optimizers for each network
    optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=0.001)
    optimizer3 = torch.optim.Adam(net3.parameters(), lr=0.001)

    # Enable GPU acceleration if available
    if torch.cuda.is_available():
        net1 = net1.cuda()
        net2 = net2.cuda()
        net3 = net3.cuda()

    # Joint training loop for all three networks
    for i in range(epoch):
        # Set all networks to training mode
        net1.train()
        net2.train()
        net3.train()
        for data in train_loader:
            x, y = data
            # Move data to GPU if available
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            # Forward pass through all networks
            outputs1 = net1(x) # Network 1 predictions
            outputs2 = net2(x) # Network 2 predictions
            outputs3 = net3(x) # Network 3 predictions

            # Complex output combination formula:
            outresults = y[:, 2]*((outputs1[:, 0]*y[:, 0] + outputs2[:, 0])/y[:, 3] + outputs3[:, 0])+y[:, 1]
            # Calculate loss against 5th column of y (index 4)
            trainloss = loss_fn(y[:, 4], outresults)

            # Backpropagation through all networks
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            trainloss.backward() # Compute gradients for all networks
            optimizer1.step() # Update network 1 parameters
            optimizer2.step() # Update network 2 parameters
            optimizer3.step() # Update network 3 parameters

            # Training progress monitoring
            total_train_step = total_train_step + 1
            if total_train_step % 10 == 0:
                print("Training step：{}, Loss: {}".format(total_train_step, trainloss.item()))

        # Evaluation phase
        net1.eval()
        net2.eval()
        net3.eval()

        with torch.no_grad(): # Disable gradient computation
            for data in test_loader:
                m, n = data
                if torch.cuda.is_available():
                    m = m.cuda()
                    n = n.cuda()

                # Generate predictions from all networks
                output1 = net1(m)
                output2 = net2(m)
                output3 = net3(m)

                # Replicate training output combination for validation
                outresult = n[:, 2] * ((output1[:, 0] * n[:, 0] + output2[:, 0])/n[:, 3] + output3[:, 0])+n[:, 1]
                # Calculate validation metrics
                testloss = loss_fn(outresult, n[:, 4])
                R2 = cal_Rsqure(n[:, 4], outresult)
                mae = cal_MAE(n[:, 4], outresult)

                print("Test Loss: {}".format(testloss.item()))
                print("Test R-squared: {}".format(R2))
                print("Test MAE: {}".format(mae))

        # Early stopping and model saving condition
        if (abs(testloss.item() - trainloss.item()) <= 0.01) and (R2 >= 0.9):
            # Save improved models
            torch.save(net1, f"netnew1.pth")
            torch.save(net2, f"netnew2.pth")
            torch.save(net3, f"netnew3.pth")
            print("Models saved successfully")
            break