import torch
import torch.nn as nn
import torch.nn.functional as F
a1=torch.Tensor([-100,-0.1,2,3])
a2=torch.Tensor([1,2,3,4])
a3=torch.Tensor([2,3,4,5])
all=torch.stack((a1,a2,a3))
# print(a)
length=a2.shape[0]
x1=a2[1:]
x2=a2[0:length-1]
# print(x1)
# print(x2)
# print(x2[::2])
size_parm={'seq_len':30,'label_len':20,'pred_len':30 }
x=size_parm['seq_len']
print(x)

