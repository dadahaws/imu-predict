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
from utils.useful_func import read_data
import math

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
class Quaternion:
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
        return self.array

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

class pose_estimation(Dataset):
    def __init__(self, root_path, flag='train'
                , data_path='data.csv' , freq=None,size=None ,batch_size=1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 200
            self.label_len = 50
            self.pred_len = 50
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # self.features = features
        # self.target = target
        # self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        ori = pd.read_csv(os.path.join(self.data_path))
        # print("check_path")
        # print(ori)
        timestamp=pd.DataFrame(ori,columns=["#timestamp"])
        quat=pd.DataFrame(ori,columns=[" q_RS_w []"," q_RS_x []"," q_RS_y []"," q_RS_z []"])
        quat_w=pd.DataFrame(ori,columns=[" q_RS_w []"])
        quat_x=pd.DataFrame(ori,columns=[" q_RS_x []"])
        quat_y=pd.DataFrame(ori,columns=[" q_RS_y []"])
        quat_z = pd.DataFrame(ori, columns=[" q_RS_z []"])
        ###此处数组化
        quat = np.array(quat.values)
        print("check_quat.arry_size")
        print(quat.shape)
        ####etc
        # border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        # border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]

        # if self.features == 'M' or self.features == 'MS':
        #     cols_data = df_raw.columns[1:]
        #     df_data = df_raw[cols_data]
        # elif self.features == 'S':
        #     df_data = df_raw[[self.target]]

        # if self.scale:
        #     #####经过此路,需要对数据做归一化
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data.values)
        #     data = self.scaler.transform(df_data.values)
        #     # print("check_data")
        #     # print(data)
        # else:
        #     data = df_data.va
        # df_stamp = quat_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # print("check_df_stamp")
        # print(df_stamp)#每一行是这些 34556 2017-06-25 23:00:00

        # data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        # print("check_time_featured")
        # print(data_stamp.shape)
        # print(data_stamp)
        self.scaler.fit(quat)

        ###通过transform　转化为tensor 的形式
        timestamp=torch.tensor(np.array(timestamp.values))
        tensor_data = self.scaler.transform(quat)
        # print("check_tensor_data")
        # print(tensor_data.shape)
        # print(type(tensor_data))
        #print(tensor_data)
        self.times = timestamp
        self.features=tensor_data
        # if self.inverse:
        #     self.data_y = df_data.values[border1:border2]
        # else:
        #     self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp


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
        # print("check_feature.shape")
        # print(self.features.shape)
        seq_quat_x = self.features[s_begin:s_end]
        seq_quat_y = self.features[r_begin:r_end]
        seq_quat_time_x = self.times[s_begin:s_end]
        seq_quat_time_y = self.times[r_begin:r_end]
        # print("check_getitem.shape")
        # print(seq_quat_time_x.shape, seq_quat_x.shape)
        return seq_quat_time_x, seq_quat_x

    def __len__(self):
        return len(self.features) - self.seq_len + 1



if __name__ == '__main__':
    root_path="/home/qzd/IMU/informer_op"
    data_path="/home/qzd/IMU/informer_op/euroc_data/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    batch_size=20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pose_dataset=pose_estimation( root_path,
    #                           flag='train',
    #                           data_path=data_path ,
    #                           freq=200,batch_size=batch_size )
    # train_loader = DataLoader(pose_dataset,
    #                       batch_size=batch_size,
    #                       shuffle=False,
    #                       num_workers=4,
    #                       drop_last=True)
    # for i, (stamp,feature) in enumerate(train_loader):
    # ##mark 代表时间戳数据
    #     print("check_i")
    #     print(i)
    #     # print("check_time")
    #     # print(stamp.shape)
    #     # print("check_feature")
    #     # print(feature.shape)


    ###以下实验分为两个部分
    ### 实验一：
    ##验证张量下的 四元数乘法运算

    q1 = Quaternion(0, 2, 4, 6)
    q2 = Quaternion(1, 3, 5, 7)
    q3 = Quaternion(2, 3, 4, 5)

    p1 = Quaternion(2, 3, 4, 5)
    p2 = Quaternion(1, 2, 3, 4)
    p3 = Quaternion(0, 1, 2, 3)
    #q2_=q2.inverse()
    mul_result_1 = q1.__mul__(p1)
    mul_result_2 = q2.__mul__(p2)
    mul_result_3 = q3.__mul__(p3)

#####将一维数组从四元数类中取出　并按照 numpy 格式储存
    arr1 = mul_result_1.return_value()
    arr2 = mul_result_2.return_value()
    arr3 = mul_result_3.return_value()

    arr1=np.expand_dims(arr1, axis=0)
    arr2 = np.expand_dims(arr2, axis=0)
    arr3 = np.expand_dims(arr3, axis=0)

    arry_mul_result=np.concatenate((arr1, arr2,arr3), axis=0)

    # print("check_mul_on_nparray")
    # print(arry_mul_result)

#####tensor下做矩阵形式的四元数乘法

    tq1 = torch.tensor(q1.return_value(),dtype=torch.float32)
    tq2 = torch.tensor(q2.return_value(),dtype=torch.float32)
    tq3 = torch.tensor(q3.return_value(),dtype=torch.float32)
    tq_all=torch.stack((tq1, tq2,tq3),dim=0)
    # print("check_tensor_array")
    # print(tq_all)

    tp1 = torch.tensor(p1.return_value(),dtype=torch.float32)
    tp2 = torch.tensor(p2.return_value(),dtype=torch.float32)
    tp3 = torch.tensor(p3.return_value(),dtype=torch.float32)
    tp_all = torch.stack((tp1, tp2, tp3),dim=0)

    tensor_mul_result=qmul(tq_all,tp_all)
    # print("check_tensor_mul_result")
    # print(tensor_mul_result)
    "以上测试四元数乘法在　numpy　与tensor 运算结果一致"

    ####实验二：　
    # 验证四元数在　numpy 与 tensor 下的求逆运算
    arr_q1 = q1.inverse().return_value()
    arr_q2 = q2.inverse().return_value()
    arr_q3 = q3.inverse().return_value()
    arr_1 = np.expand_dims(arr_q1, axis=0)
    arr_2 = np.expand_dims(arr_q2, axis=0)
    arr_3 = np.expand_dims(arr_q3, axis=0)

    numpy_inverse_result = np.concatenate((arr_1, arr_2, arr_3), axis=0)
    print("check_numpy_inverse_result")
    print(numpy_inverse_result)
    print("check_tensor_result_on_inverse")
    tensor_q_inverse_result=q_inverse(tq_all)
    print(tensor_q_inverse_result)

    "以上测试四元数求逆（非单位四元数）在　numpy　与tensor 运算结果一致"

    ####实验三：　验证模运算　
    q_norm=q_norm(tq_all)

    arr_q1 = q1.norm().return_value()
    arr_q2 = q2.norm().return_value()
    arr_q3 = q3.norm().return_value()
    arr_1 = np.expand_dims(arr_q1, axis=0)
    arr_2 = np.expand_dims(arr_q2, axis=0)
    arr_3 = np.expand_dims(arr_q3, axis=0)
    arry_norm_result = np.concatenate((arr_1 ,arr_2 , arr_3), axis=0)

    print("check_norm_on_numpy")
    print(arry_norm_result)
    print("check_norm_on_tensor")
    print(q_norm)
    "以上测试四元数求模（非单位四元数）在　numpy　与tensor 运算结果一致"










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
