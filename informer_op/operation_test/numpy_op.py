import numpy as np

path="./test.txt"

a=np.array([1,2,3,4,5,6],dtype=np.float32)
np.set_printoptions(suppress=True)###显示所有小数
np.set_printoptions(precision=4)   #设精度
np.savetxt(path,a, fmt='%.04f',delimiter=" ")   #保留4位小数
print(a)