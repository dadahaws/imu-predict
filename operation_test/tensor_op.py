import torch
import torch.nn as nn
import torch.nn.functional as F
t1 = torch.tensor([[2, 1, 3], [4, 5, 9]],dtype=torch.float32)
a1=torch.tensor([2, 1, 3],dtype=torch.float32)
t2 = t1.transpose(0, 1).contiguous()
print(t1)
print(t2)
t2[0,1]=2222
print(t1)
print(t2)
m = nn.Softmax(dim=0)###在行上相加等于１
n = nn.Softmax(dim=1)###在列上相加等于１

i=F.normalize(t1,p=2,dim=0)####行上的平方和为1
j=F.normalize(t1,p=2,dim=1)####列上的平方和为1

# b = torch.Tensor([[[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]])
# print(b.shape)
# print(b)
# c=torch.nn.functional.normalize(b, dim=2)
# print(c)

## 截断操作　torch.clamp(input, min, max, out=None)
s=a1.expand(6, -1)
print(s)

x = torch.tensor([1, 2, 3])
print(x.shape)


#####矩阵乘法
#terms=torch.bmm(q.view(-1, 1, 4), tmp)
#terms = torch.matmul(q.view(-1, 1, 4), tmp)