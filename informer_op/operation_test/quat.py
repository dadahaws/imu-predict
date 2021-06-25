import numpy as np
import quaternionmath as quat
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler
import math
from utils.useful_func import *
from torch.autograd import Variable

# class Quaternion:
#     def __init__(self,array):
#         self.w=0
#         self.x=array[0]
#         self.y=array[1]
#         self.z=array[2]
#         self.array=array
#     def toArray():
#         return [x,y,z,w]
#     def __add__(self,quaternion):
#         result=Quaternion(self.array)
#         result.w+=quaternion.w
#         result.x+=quaternion.x
#         result.y+=quaternion.y
#         result.z+=quaternion.z
#         return result
#     def __sub__(self,quaternion):
#         result=Quaternion(self.array)
#         result.w-=quaternion.w
#         result.x-=quaternion.x
#         result.y-=quaternion.y
#         result.z-=quaternion.z
#         return result
#     def multiplication(self,quaternion):
#         result=Quaternion(self.array)
#         result.w=self.w*quaternion.w-self.x*quaternion.x-self.y*quaternion.y-self.z*quaternion.z
#         result.x=self.w*quaternion.x+self.x*quaternion.w+self.y*quaternion.z-self.z*quaternion.y
#         result.y=self.w*quaternion.y-self.x*quaternion.z+self.y*quaternion.w+self.z*quaternion.x
#         result.z=self.w*quaternion.z+self.x*quaternion.y-self.y*quaternion.x+self.z*quaternion.w
#         return result
#     def divides(quaternion):
#         result=Quaternion(self.array)
#         return result.multiplication(quaternion.inverse());
#     def mod(self):
#         return pow((pow(self.x,2)+pow(self.y,2)+pow(self.z,2)+pow(self.w,2)),1/2)
#     def star(self):
#         result=Quaternion(self.array)
#         result.w=self.w
#         result.x=-self.x
#         result.y=-self.y
#         result.z=-self.z
#         return result
#     def inverse(self):
#         result=Quaternion(self.array)
#         moder=self.mod()
#         result.w/=moder
#         result.x/=moder
#         result.y/=moder
#         result.z/=moder
#         return result
#     def __str__(self):
#         return str(self.x)+"i "+str(self.y)+"j "+str(self.z)+"k "+str(self.w)

# def quaternion_multiply(quaternion1, quaternion0):
#     w0, x0, y0, z0 = quaternion0
#     w1, x1, y1, z1 = quaternion1
#     return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
#                      x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
#                      -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
#                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
class Quaternion_arr:   ###numpy下测试
    def __init__(self, s, x, y, z):
        """构造函数"""
        self.s = s
        self.x = x
        self.y = y
        self.z = z
        self.vector = [x, y, z]
        self.all = [s, x, y, z]
        self.array=np.array([s, x, y, z])

    def __str__(self):
        """输出操作重载"""
        # # op = [" ", "i ", "j ", "k"]
        # q = self.all.copy()
        # result = ""
        # for i in range(4):
        #     # if q[i] < -1e-8 or q[i] > 1e-8:
        #         result = result + str(round(q[i], 6))+" "
        # if result == "":
        #     return "0"
        # else:
        #     return result
        for i in range(4):
            if self.array[i] > -1e-12 and self.array[i] < 1e-12:
                self.array[i]=0
        return str(self.array)

    def __add__(self, quater):
        """加法运算符重载"""
        q = self.all.copy()
        for i in range(4):
            q[i] += quater.all[i]
        return Quaternion(q[0], q[1], q[2], q[3])

    def __sub__(self, quater):
        """减法运算符重载"""
        q = self.all.copy()
        for i in range(4):
            q[i] -= quater.all[i]
        return Quaternion(q[0], q[1], q[2], q[3])

    def __mul__(self, quater):
        """乘法运算符重载"""
        q = self.array.copy()

        p = quater.array.copy()
        s = q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3]
        x = q[0] * p[1] + q[1] * p[0] + q[2] * p[3] - q[3] * p[2]
        y = q[0] * p[2] - q[1] * p[3] + q[2] * p[0] + q[3] * p[1]
        z = q[0] * p[3] + q[1] * p[2] - q[2] * p[1] + q[3] * p[0]
        return Quaternion(s, x, y, z)

    def divide(self, quaternion):
        """右除"""
        result = self * quaternion.inverse()
        return result

    def modpow(self):
        """模的平方"""
        q = self.all
        return sum([i ** 2 for i in q])

    def mod(self):
        """求模"""
        return pow(self.modpow(), 1 / 2)

    def conj(self):
        """转置"""
        q = self.all.copy()
        for i in range(1, 4):
            q[i] = -q[i]
        return Quaternion(q[0], q[1], q[2], q[3])

    def inverse(self):
        """求逆"""
        q = self.all.copy()
        mod = self.modpow()
        for i in range(4):
            q[i] /= mod
        return Quaternion(q[0], -q[1], -q[2], -q[3])
    def norm(self):
        """转化为单位四元数"""
        tmp=self.all.copy()
        x=self.mod()
        return Quaternion(tmp[0]/x,tmp[1]/x,tmp[2]/x,tmp[3]/x)
    def shape(self):
        return self.array.shape
    def return_value(self):
        return self.array####numpy###nupnumpy

####tensor下测试
def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    print(type(r))
    print(type(q))
    terms = torch.bmm(r.view(-1, 4, 1).float(), q.view(-1, 1, 4).float())

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def q_inverse(q):
    assert q.shape[-1] == 4
    q_length=q.shape[0]
    original_shape = q.shape

    t_q=q.view(-1, 1, 4)
    # print("t_q")
    # print(t_q)

    ###求q　的摸的平方
    q_norm=torch.pow(q, 2, out=None)
    # print("check_terms_norm_2")
    # print(q_norm.shape)
    # print(q_norm)
    q_norm=torch.sum(q_norm,dim=1)
    # print("check_terms_norm_sum")
    # print(q_norm.shape)
    # print(q_norm)
    q_norm= q_norm.expand(4, -1).transpose(0,1).unsqueeze(1)##将一维向量扩展
    # print("check_terms_norm_expand")
    # print(q_norm.shape)
    # print(q_norm)
    q_norm=torch.reciprocal(q_norm, out=None)
    # print("check_terms_norm_inverse")
    # print(terms_norm)


    tmp=torch.tensor([1,-1,-1,-1],dtype=torch.float64)   ###取共轭
    tmp=tmp.expand(q_length, -1).contiguous().unsqueeze(1)
    # print("check_tmp")
    # print(tmp)
    terms = torch.mul(t_q, tmp)
    # print("check_terms.shape")
    # print(terms.shape)
    # print("check_terms")
    # print(terms)
    # print("check_terms.shape")
    # print(terms.shape)
    # print("check_terms.value")
    # print(terms)

    q=torch.mul(terms ,q_norm)  ###q*/|| q ||^2
    #q=torch.clamp(q,1e-8,1e5)
    # print("check_q.shape")
    # print(q.shape)
    # print("check_q")
    # print(q)
    return q

def q_norm(q):
    assert q.shape[-1] == 4
    q=q.view(-1, 1, 4)
    q=F.normalize(q, dim=2)
    return q

def quat_distance(q1,q2):
    q2 = q_inverse(q2)
    delta = qmul(q2, q1)
    return delta
def loss_func_quat_norm(out,var_y,parm1,iteration):
    if iteration%1000==0:
        print(out)
    loss1 = criterion1(out, var_y)
    delta_one = torch.ones(batch_size).float()
    # print("check_out.shape")
    # print(out.shape)
    delta = torch.sum(torch.pow(out, 2), dim=1).float()
    loss2 = criterion2(delta, delta_one)
    # loss=criterion.forward(out,var_y)
    # print("check_loss.shape")
    # print(loss)
    all_loss = loss1 + loss2
    return all_loss

class quat_loss(nn.Module):
    def __init__(self):
        super(quat_loss,self).__init__()


    def qmul(q, r):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
        Returns q*r as a tensor of shape (*, 4).
        """
        assert q.shape[-1] == 4
        assert r.shape[-1] == 4

        original_shape = q.shape

        # Compute outer product
        terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.stack((w, x, y, z), dim=1).view(original_shape)

    def q_inverse(q):
        assert q.shape[-1] == 4
        q_length = q.shape[0]
        original_shape = q.shape

        t_q = q.view(-1, 1, 4)

        ###求q　的摸的平方
        q_norm = torch.pow(q, 2, out=None)
        q_norm = torch.sum(q_norm, dim=1)

        q_norm = q_norm.expand(4, -1).transpose(0, 1).unsqueeze(1)  ##将一维向量扩展
        q_norm = torch.reciprocal(q_norm, out=None)

        tmp = torch.tensor([1, -1, -1, -1], dtype=torch.float64)  ###取共轭
        tmp = tmp.expand(q_length, -1).contiguous().unsqueeze(1)
        terms = torch.mul(t_q, tmp)
        q = torch.mul(terms, q_norm)  ###q*/|| q ||^2

        return q

    def q_norm(q):
        assert q.shape[-1] == 4
        q = q.view(-1, 1, 4)
        q = F.normalize(q, dim=2)
        return q

    def quat_distance(q1, q2):
        q2 = q_inverse(q2)
        delta = qmul(q2, q1)
        return delta

    def forward(self,x1,x2):
        distance=quat_distance(x1, x2)
        # print(x1.shape)#(w,h)
        # print(x1)
        # print(x2.shape)  # (w,h)
        # print(x2)
        # print("check_distance")
        # print(distance.shape)
        delta_one=torch.zeros_like(distance)
        delta_one[:,:,0]=-1.0
        result=torch.add(distance,delta_one)
        # print(result)
        loss=torch.mean(result)
        # print("check_loss")
        # print(loss)
        return loss

#####旋转变量上数据
class pose_estimation(Dataset):
    def __init__(self, root_path,imu_data_path,delta_data_path,
                 flag='train',freq=None,size=None ,batch_size=1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 30       ###此处设定返回的片段长度
            self.label_len = 20
            self.pred_len = 30
        else:
            self.seq_len = size['seq_len']
            self.label_len = size['label_len']
            self.pred_len = size['pred_len']
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # self.features = features
        # self.target = target
        # self.freq = freq

        self.root_path = root_path
        self.gt_data_path = gt_data_path
        self.imu_data_path=imu_data_path
        self.delta_data_path=delta_data_path
        self.__read_data__()

    def __read_data__(self):
        delta_quat=read_delta_quat_txt(self.delta_data_path)
        t, w, a = read_imu_data_txt(self.imu_data_path)

        self.scaler = StandardScaler()##获得均值，方差信息
        self.scaler.fit(w)

        ###通过transform　转化为tensor 的形式
        t=torch.tensor(t)
        tensor_data = self.scaler.transform(w)###数据归一化

        # print(type(tensor_data))
        #print(tensor_data)
        self.times = t
        self.features=tensor_data
        self.targets=delta_quat
        print("check_feature.shape")
        print(self.features.shape)
        print("check_target.shape")
        print(self.targets.shape)

    def __getitem__(self, index):
        s_begin = index
        # print("check_s_begin")
        # print(s_begin)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len##informer 此处使用　不知道是什么意思
        r_end = r_begin + self.label_len + self.pred_len
        # print("check_s_index")
        # print(s_end-s_begin)
        # print("check_r_index")
        # print(r_end-r_begin)

        time = self.times[s_begin:s_end]
        seq_w = self.features[s_begin:s_end]
        delta_q=self.targets[s_begin]
        # print("check_getitem.shape")
        # print(seq_quat.shape, time.shape)
        return time,  seq_w ,delta_q

    def __len__(self):
        print("检查索引上限")####所有样本数量
        print(len(self.features) - self.seq_len )
        return len(self.features) - self.seq_len

class BiLSTMNet(nn.Module):

    def __init__(self, input_size):
        super(BiLSTMNet, self).__init__()
        self.rnn = nn.LSTM(
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
            nn.Linear(128, 4)
        )
        self.h0=torch.randn(2, 32, 64).float()

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, self.h0)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        return out

if __name__ == '__main__':
    root_path="/home/qzd/IMU/informer_op"
    gt_data_path="/home/qzd/IMU/informer_op/test_dataset/gt_data.csv"
    imu_data_path="/home/qzd/IMU/informer_op/test_dataset/real_imu.txt"
    delta_data_path="/home/qzd/IMU/informer_op/test_dataset/delta_q_data.txt"
    batch_size=32
    size_parm={'seq_len':30,'label_len':20,'pred_len':30 }
    net = GRUNet(3)
    #criterion = nn.MSELoss()
    criterion1 = quat_loss()
    criterion2 =nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pose_dataset=pose_estimation( root_path=root_path,
                                imu_data_path=imu_data_path,
                                  delta_data_path=delta_data_path,
                                flag='train',
                                freq=200,
                                size=size_parm,
                                batch_size=batch_size )
    train_loader =DataLoader(
        pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)
    # print(train_loader.__len__())
    # print("check_length_train_loader")
    # print(len(train_loader))
    for e in range(1000):
        for i, (time,feature,target) in enumerate(train_loader):###此处ｉ受到batch影响
    ##mark 代表时间戳数据
        # print("check_i")
        # print(i)
        # print("check_time")
        # print(time.shape)
        # print("check_feature")
        # print(feature.shape)
        # print("check_target")
        # print(target.shape)
            var_x = Variable(feature)
            var_y = Variable(target)
            out = net(var_x.float())
            # print("check_out.shape")
            # print(out.shape)
            # print("check_vary.shape")
            # print(var_y.shape)
            loss=loss_func_quat_norm(out,var_y,parm1=10,iteration=i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100==0 :
                print('Epoch: {}, iter : {} Loss: {:.5f}'.format(e + 1,i, loss.item()))
            #if (e + 1) % 100 == 0:  # 每 100 次输出结果
                #torch.save(obj=net.state_dict(), f='models/lstmnetpro_gru_%d.pth' % (e + 1))

    ###以下实验分为两个部分
    ### 实验一：
    ##验证张量下的 四元数乘法运算

#     q1 = Quaternion(0, 2, 4, 6)
#     q2 = Quaternion(1, 3, 5, 7)
#     q3 = Quaternion(2, 3, 4, 5)
#
#     p1 = Quaternion(2, 3, 4, 5)
#     p2 = Quaternion(1, 2, 3, 4)
#     p3 = Quaternion(0, 1, 2, 3)
#     #q2_=q2.inverse()
#     mul_result_1 = q1.__mul__(p1)
#     mul_result_2 = q2.__mul__(p2)
#     mul_result_3 = q3.__mul__(p3)
#
# #####将一维数组从四元数类中取出　并按照 numpy 格式储存
#     arr1 = mul_result_1.return_value()
#     arr2 = mul_result_2.return_value()
#     arr3 = mul_result_3.return_value()
#
#     arr1=np.expand_dims(arr1, axis=0)
#     arr2 = np.expand_dims(arr2, axis=0)
#     arr3 = np.expand_dims(arr3, axis=0)
#
#     arry_mul_result=np.concatenate((arr1, arr2,arr3), axis=0)
#
#     # print("check_mul_on_nparray")
#     # print(arry_mul_result)
#
# #####tensor下做矩阵形式的四元数乘法
#
#     tq1 = torch.tensor(q1.return_value(),dtype=torch.float32)
#     tq2 = torch.tensor(q2.return_value(),dtype=torch.float32)
#     tq3 = torch.tensor(q3.return_value(),dtype=torch.float32)
#     tq_all=torch.stack((tq1, tq2,tq3),dim=0)
#     # print("check_tensor_array")
#     # print(tq_all)
#
#     tp1 = torch.tensor(p1.return_value(),dtype=torch.float32)
#     tp2 = torch.tensor(p2.return_value(),dtype=torch.float32)
#     tp3 = torch.tensor(p3.return_value(),dtype=torch.float32)
#     tp_all = torch.stack((tp1, tp2, tp3),dim=0)
#
#     tensor_mul_result=qmul(tq_all,tp_all)
#     # print("check_tensor_mul_result")
#     # print(tensor_mul_result)
#     "以上测试四元数乘法在　numpy　与tensor 运算结果一致"
#
#     ####实验二：　
#     # 验证四元数在　numpy 与 tensor 下的求逆运算
#     arr_q1 = q1.inverse().return_value()
#     arr_q2 = q2.inverse().return_value()
#     arr_q3 = q3.inverse().return_value()
#     arr_1 = np.expand_dims(arr_q1, axis=0)
#     arr_2 = np.expand_dims(arr_q2, axis=0)
#     arr_3 = np.expand_dims(arr_q3, axis=0)
#
#     numpy_inverse_result = np.concatenate((arr_1, arr_2, arr_3), axis=0)
#     print("check_numpy_inverse_result")
#     print(numpy_inverse_result)
#     print("check_tensor_result_on_inverse")
#     tensor_q_inverse_result=q_inverse(tq_all)
#     print(tensor_q_inverse_result)
#
#     "以上测试四元数求逆（非单位四元数）在　numpy　与tensor 运算结果一致"
#
#     ####实验三：　验证模运算　
#     q_norm=q_norm(tq_all)
#
#     arr_q1 = q1.norm().return_value()
#     arr_q2 = q2.norm().return_value()
#     arr_q3 = q3.norm().return_value()
#     arr_1 = np.expand_dims(arr_q1, axis=0)
#     arr_2 = np.expand_dims(arr_q2, axis=0)
#     arr_3 = np.expand_dims(arr_q3, axis=0)
#     arry_norm_result = np.concatenate((arr_1 ,arr_2 , arr_3), axis=0)
#
#     print("check_norm_on_numpy")
#     print(arry_norm_result)
#     print("check_norm_on_tensor")
#     print(q_norm)
#     "以上测试四元数求模（非单位四元数）在　numpy　与tensor 运算结果一致"










# x1=Quaternion(1,2,3,4)
# x2=Quaternion(1,2,3,4)
#
# x3=x1*x2
# print(x3)
#
# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# print(input)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(target)
# output = loss(input, target)
# print(output)
