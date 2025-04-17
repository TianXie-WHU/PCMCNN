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
    y = read_xlrd(excelFile=excelFile1, sheet_name='e2')

x1 = torch.FloatTensor(x)
y1 = torch.FloatTensor(y)

x = torch.Tensor(x)
y = torch.Tensor(y)
dataset = Data.TensorDataset(x, y)
train_data, test_data=random_split(dataset, [round(0.8*x.shape[0]),round(0.2*x.shape[0])])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10000, shuffle=True, num_workers=0,)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=True, num_workers=0,)

net = Net()

if torch.cuda.is_available():
    net = net.cuda()


loss_fn = nn.MSELoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10000
for i in range(epoch):
    net.train()
    for data in train_loader:
        x, y = data
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        outputs = net(x)
        trainloss = loss_fn(outputs, y)
        optimizer.zero_grad()
        trainloss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, trainloss.item()))

        net.eval()
        with torch.no_grad():
            for test in test_loader:
                m, n = data
                if torch.cuda.is_available():
                    m = m.cuda()
                    n = n.cuda()
                output = net(m)
                testloss = loss_fn(output, n)
                R2 = cal_Rsqure(n, output)
                mae = cal_MAE(n, output)
                print("整体测试集上的Loss: {}".format(testloss.item()))
                print("整体测试集上的正确率: {}".format(R2))
                print("整体测试集上的MAE: {}".format(mae))

    if(abs(testloss.item() - trainloss.item()) <= 0.01) and (R2 >= 0.9):
        torch.save(net, "net2.pth")
        print("模型已保存")
        break