import torch
from utils.data_loader import read_xlrd
import torch.utils.data as Data
from torch.utils.data import random_split
from utils.metrics import cal_Rsqure, cal_MAE
from utils.model import Net
from torch import nn

if __name__ == '__main__':
    excelFile1 = 'data/TGLAND.xlsx'
    x = read_xlrd(excelFile=excelFile1, sheet_name='water')
    y = read_xlrd(excelFile=excelFile1, sheet_name='result')

x = torch.Tensor(x)
y = torch.Tensor(y)
dataset = Data.TensorDataset(x, y)
train_data, test_data=random_split(dataset, [round(0.8*x.shape[0]),round(0.2*x.shape[0])])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10000, shuffle=True, num_workers=0,)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=True, num_workers=0,)
net1 = Net()
net2 = Net()
net3 = Net()
net1 = torch.load("net1.pth")
net2 = torch.load("net2.pth")
net3 = torch.load("net3.pth")

loss_fn = nn.MSELoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()


# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 100000
optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(net2.parameters(), lr=0.001)
optimizer3 = torch.optim.Adam(net3.parameters(), lr=0.001)
if torch.cuda.is_available():
    net1 = net1.cuda()
    net2 = net2.cuda()
    net3 = net3.cuda()

for i in range(epoch):
    net1.train()
    net2.train()
    net3.train()
    for data in train_loader:
        x, y = data
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        outputs1 = net1(x)
        outputs2 = net2(x)
        outputs3 = net3(x)
        outresults = y[:, 2]*((outputs1[:, 0]*y[:, 0] + outputs2[:, 0])/y[:, 3] + outputs3[:, 0])+y[:, 1]
        trainloss = loss_fn(y[:, 4], outresults)

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        trainloss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, trainloss.item()))

        net1.eval()
        net2.eval()
        net3.eval()

        with torch.no_grad():
            for data in test_loader:
                m, n = data
                if torch.cuda.is_available():
                    m = m.cuda()
                    n = n.cuda()
                output1 = net1(m)
                output2 = net2(m)
                output3 = net3(m)
                outresult = n[:, 2] * ((output1[:, 0] * n[:, 0] + output2[:, 0])/n[:, 3] + output3[:, 0])+n[:, 1]
                testloss = loss_fn(outresult, n[:, 4])
                R2 = cal_Rsqure(n[:, 4], outresult)
                mae = cal_MAE(n[:, 4], outresult)
                print("整体测试集上的Loss: {}".format(testloss.item()))
                print("整体测试集上的r2: {}".format(R2))
                print("整体测试集上的MAE: {}".format(mae))
    if (abs(testloss.item() - trainloss.item()) <= 0.01) and (R2 >= 0.9):
        torch.save(net1, f"netnew1.pth")
        torch.save(net2, f"netnew2.pth")
        torch.save(net3, f"netnew3.pth")
        print("模型已保存")
        break