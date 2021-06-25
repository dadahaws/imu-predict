import numpy as np
import torch
import math
path="./test.txt"

a=np.array([1,2,3,4,5,6],dtype=np.float32)
np.set_printoptions(suppress=True)###显示所有小数
np.set_printoptions(precision=4)   #设精度
np.savetxt(path,a, fmt='%.04f',delimiter=" ")   #保留4位小数
def clone_a(num):
    return num**2,num+2
# x=2
# att=[]
# a=[clone_a for l in range(4)]
# for fun in a:
#     print(x)
#     x,re2=fun(x)
#     print(x)
#     att.append(re2)

# print(a)
# a=None
# b=[5] if a else None
# print(b)
for l in range(3):
    print(l)