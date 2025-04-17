import torch
from utils.data_loader import read_xlrd
import torch.utils.data as Data
import pandas as pd
from utils.model import Net

if __name__ == '__main__':
    excelFile1 = 'data/stationtotal.xlsx'
    x = read_xlrd(excelFile=excelFile1, sheet_name='water')
    y = read_xlrd(excelFile=excelFile1, sheet_name='result')

x1 = torch.FloatTensor(x)
y1 = torch.FloatTensor(y)

dataset = Data.TensorDataset(x1, y1)
test_loader = torch.utils.data.DataLoader(dataset,batch_size=3185,shuffle=False)
net1 = Net()
net2 = Net()
net3 = Net()
net1 = torch.load(f"netnew1.pth",map_location='cpu')
net2 = torch.load(f"netnew2.pth",map_location='cpu')
net3 = torch.load(f"netnew3.pth",map_location='cpu')
if torch.cuda.is_available():
    net1 = net1.cuda()
    net2 = net2.cuda()
    net3 = net3.cuda()

for data in test_loader:
    m, n = data
    if torch.cuda.is_available():
        m = m.cuda()
        n = n.cuda()
    output1 = net1(m)
    output2 = net2(m)
    output3 = net3(m)
    outresult = n[:, 2] * ((output1[:, 0] * n[:, 0] + output2[:, 0]) / n[:, 3] + output3[:, 0]) + n[:, 1]

array_1 = outresult.detach().cpu().numpy()
data = pd.DataFrame(array_1)
array_2 = n[:, 4].cpu().numpy()
data2 = pd.DataFrame(array_2)
writer = pd.ExcelWriter('stationresult.xlsx')
data.to_excel(writer, 'predict-LST', float_format='%.6f', header=False, index=False)
data2.to_excel(writer, 'label-LST', float_format='%.6f', header=False, index=False)
writer.close()
