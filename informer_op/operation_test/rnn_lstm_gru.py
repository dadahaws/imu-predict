import torch
from torch.autograd import Variable
import torch.nn as nn

#####  RNN   ####
rnn = nn.RNN(input_size=4,hidden_size=50,num_layers=2)##input_size：输入x 的特征维度
input_data = Variable(torch.randn(100,1,4))  #seq,batch,feature
#如果传入网络时，不特别注明隐状态，那么输出的隐状态默认参数全是0
h_0 = Variable(torch.randn(2,1,50))  #layer*direction,batch,hidden_size
output,h_t = rnn(input_data,h_0)

# print(output.size())  #seq,batch,hidden_size
# print(h_t.size())   #layer*direction,batch,hidden_size
# print(rnn.weight_ih_l0.size())

##### LSTM  #####
#定义网络
lstm = nn.LSTM(input_size=20,hidden_size=50,num_layers=2)
#输入变量
input_data = Variable(torch.randn(100,32,20))
#初始隐状态
h_0 = Variable(torch.randn(2,32,50))
#输出记忆细胞
c_0 = Variable(torch.randn(2,32,50))
#输出变量
output,(h_t,c_t) = lstm(input_data,(h_0,c_0))
# print(output.size())
# print(h_t.size())
# print(c_t.size())
# #参数大小为(50x4,20),是RNN的四倍
# print(lstm.weight_ih_l0)
# print(lstm.weight_ih_l0.size())

####  gru  ####
gru = nn.GRU(input_size=20,hidden_size=50,num_layers=2)
#输入变量
input_data = Variable(torch.randn(100,32,20))
#初始隐状态
h_0 = Variable(torch.randn(2,32,50))
#输出变量
output,(h_n,c_n) = gru(input_data)  #lstm(input_data,h_0) 不定义初始隐状态默认为0
# print(output.size())
# print(h_n.size())
# print(gru.weight_ih_l0)
# print(gru.weight_ih_l0.size())


class GRUNet(nn.Module):

    def __init__(self, input_size):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        print(out.shape)
        return out
if __name__ == '__main__':

    net = GRUNet(3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)
    # start training
    for e in range(1000):
        for i, (X, y) in enumerate(train_loader):
            var_x = Variable(X)
            var_y = Variable(y)
            # forward
            out = net(var_x)
            loss = criterion(out, var_y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
            if (e + 1) % 100 == 0:  # 每 100 次输出结果
                torch.save(obj=net.state_dict(), f='models/lstmnetpro_gru_%d.pth' % (e + 1))

    torch.save(obj=net.state_dict(), f="models/lstmnet_gru_1000.pth")