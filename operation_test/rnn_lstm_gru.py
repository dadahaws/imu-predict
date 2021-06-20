import torch
from torch.autograd import Variable
import torch.nn as nn

#####  RNN   ####
rnn = nn.RNN(input_size=4,hidden_size=50,num_layers=2)##input_size：输入x 的特征维度
input_data = Variable(torch.randn(100,1,4))  #seq,batch,feature
#如果传入网络时，不特别注明隐状态，那么输出的隐状态默认参数全是0
h_0 = Variable(torch.randn(2,1,50))  #layer*direction,batch,hidden_size
output,h_t = rnn(input_data,h_0)

print(output.size())  #seq,batch,hidden_size
print(h_t.size())   #layer*direction,batch,hidden_size
print(rnn.weight_ih_l0.size())

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
print(output.size())
print(h_t.size())
print(c_t.size())
#参数大小为(50x4,20),是RNN的四倍
print(lstm.weight_ih_l0)
print(lstm.weight_ih_l0.size())

####  gru  ####
gru = nn.GRU(input_size=20,hidden_size=50,num_layers=2)
#输入变量
input_data = Variable(torch.randn(100,32,20))
#初始隐状态
h_0 = Variable(torch.randn(2,32,50))
#输出变量
output,(h_n,c_n) = gru(input_data)  #lstm(input_data,h_0) 不定义初始隐状态默认为0
print(output.size())
print(h_n.size())
print(gru.weight_ih_l0)
print(gru.weight_ih_l0.size())