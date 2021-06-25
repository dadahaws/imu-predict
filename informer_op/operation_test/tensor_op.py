import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def PositionalEmbedding():
    d_model=64
    max_len=200
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False
    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    # print(position)
    # print(position.shape)
    # print(div_term)
    # print(div_term.shape)
    print(pe)
    print(pe.shape)
    return  pe

class TokenEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                               kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
t1 = torch.tensor([[2, 1, 3], [4, 5, 9]],dtype=torch.float32)
a1=torch.tensor([2, 1, 3],dtype=torch.float32)
t2 = t1.transpose(0, 1).contiguous()
# print(t1)
# print(t2)
# t2[0,1]=2222
# print(t1)
# print(t2)
m = nn.Softmax(dim=0)###在行上相加等于１
n = nn.Softmax(dim=1)###在列上相加等于１

i=F.normalize(t1,p=2,dim=0)####行上的平方和为1
j=F.normalize(t1,p=2,dim=1)####列上的平方和为1

b1 = torch.Tensor([[[1,2,3], [4,5,6]], [[7,8,9], [0,1,2]]])
b2 = torch.Tensor([[[0.1,0.2,0.3], [0.4,0.5,0.6]], [[0.7,0.8,0.9], [0,0.1,0.2]]])
b3=torch.cat([b1,b2],dim=1)
# print(b3.shape)
# print(b3)

b=torch.rand(7,10,4)
a=torch.zeros_like(b)
a[:,:,0]=1.0
# print(a.shape)
# print(a)
# c=torch.nn.functional.normalize(b, dim=2)



## 截断操作　torch.clamp(input, min, max, out=None)
# s=a1.expand(6, -1)
# print(s)

#####矩阵乘法
#terms=torch.bmm(q.view(-1, 1, 4), tmp)
#terms = torch.matmul(q.view(-1, 1, 4), tmp)




# token_x=TokenEmbedding(3,64)
# x=torch.rand(3,100,7)
# result1=token_x(x)
# print(result.shape)
# result2=PositionalEmbedding()
# result=result1+result2
x=torch.nn.LayerNorm(7)
print(x)