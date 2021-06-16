import numpy as np
import quaternionmath as quat
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
class Quaternion:
    def __init__(self, s, x, y, z):
        """构造函数"""
        self.s = s
        self.x = x
        self.y = y
        self.z = z
        self.vector = [x, y, z]
        self.all = [s, x, y, z]

    def __str__(self):
        """输出操作重载"""
        op = [" ", "i ", "j ", "k"]
        q = self.all.copy()
        result = ""
        for i in range(4):
            if q[i] < -1e-8 or q[i] > 1e-8:
                result = result + str(round(q[i], 4)) + op[i]
        if result == "":
            return "0"
        else:
            return result

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
        q = self.all.copy()
        p = quater.all.copy()
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
        q = self.all()
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


class pose_estimation(Dataset):
    def __init__(self, root_path, flag='train'
                , data_path='data.csv' , freq=200, ):
        # size [seq_len, label_len, pred_len]
        # info
        # if size == None:
        #     self.seq_len = 24 * 4 * 4
        #     self.label_len = 24 * 4
        #     self.pred_len = 24 * 4
        # else:
        #     self.seq_len = size[0]
        #     self.label_len = size[1]
        #     self.pred_len = size[2]
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
        ori = pd.read_csv(os.path.join(self.data_path))
        print("check_path")
        print(ori)
        timestamp=pd.DataFrame(ori,columns=["#timestamp"])
        quat=pd.DataFrame(ori,columns=[" q_RS_w []", " q_RS_x []"," q_RS_y []"," q_RS_z []"])
        print("check_timestamp")
        print(timestamp.values)
        print(timestamp.shape)
        print("check_quat")
        print(quat.values)
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
        self.times = timestamp
        self.features=quat
        # if self.inverse:
        #     self.data_y = df_data.values[border1:border2]
        # else:
        #     self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp


    def __getitem__(self, index):


        return self.times,self.features

    def __len__(self):
        return len(self.times)


if __name__ == '__main__':
    root_path="/home/qzd/IMU/informer_op"
    data_path="/home/qzd/IMU/informer_op/euroc_data/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pose_dataset=pose_estimation( root_path,
                              flag='train',
                              data_path=data_path ,
                              freq=200 )
    train_loader = DataLoader(pose_dataset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True)
    for i, (stamp,feature) in enumerate(train_loader):
    ##mark 代表时间戳数据
        print("check_time")
        print(stamp)
        print("check_feature")
        print(feature)










x1=Quaternion(1,2,3,4)
x2=Quaternion(1,2,3,4)

x3=x1*x2
print(x3)

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print(input)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
output = loss(input, target)
print(output)
