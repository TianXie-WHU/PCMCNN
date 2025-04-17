import torch
from utils.data_loader import read_xlrd
import torch.utils.data as Data
import pandas as pd
from utils.model import Net

if __name__ == '__main__':
    # Data loading configuration
    excelFile1 = 'data/stationtotal.xlsx'
    # Load input features from 'water' sheet
    x = read_xlrd(excelFile=excelFile1, sheet_name='water')
    # Load target values from 'result' sheet
    y = read_xlrd(excelFile=excelFile1, sheet_name='result')

    # Convert to PyTorch tensors with explicit float32 typing
    x1 = torch.FloatTensor(x)  # Input features tensor
    y1 = torch.FloatTensor(y)  # Target values tensor

    # Create dataset and loader for full-batch inference
    dataset = Data.TensorDataset(x1, y1)
    test_loader = torch.utils.data.DataLoader(dataset,batch_size=3185,shuffle=False)

    # Model initialization and loading
    net1 = Net() # Placeholder initialization
    net2 = Net()
    net3 = Net()
    # Load well-trained models
    net1 = torch.load(f"netnew1.pth")
    net2 = torch.load(f"netnew2.pth")
    net3 = torch.load(f"netnew3.pth")

    # Enable GPU acceleration if available
    if torch.cuda.is_available():
        net1 = net1.cuda()
        net2 = net2.cuda()
        net3 = net3.cuda()

    for data in test_loader:
        m, n = data # Unpack batch data
        # Device configuration
        if torch.cuda.is_available():
            m = m.cuda()
            n = n.cuda()

        # Model predictions
        output1 = net1(m)  # Network 1 output
        output2 = net2(m)  # Network 2 output
        output3 = net3(m)  # Network 3 output

        # Output combination
        outresult = n[:, 2] * ((output1[:, 0] * n[:, 0] + output2[:, 0]) / n[:, 3] + output3[:, 0]) + n[:, 1]

        # Convert results to numpy arrays
        array_1 = outresult.detach().cpu().numpy()  # Predictions
        array_2 = n[:, 4].cpu().numpy()  # Ground truth

        # Create DataFrames for Excel export
        data = pd.DataFrame(array_1)
        data2 = pd.DataFrame(array_2)

        # Save results to Excel with precision formatting
        writer = pd.ExcelWriter('stationresult.xlsx')
        data.to_excel(writer, 'predict-LST', float_format='%.6f', header=False, index=False)
        data2.to_excel(writer, 'label-LST', float_format='%.6f', header=False, index=False)
        writer.close()
