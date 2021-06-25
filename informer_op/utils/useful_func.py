import pandas as pd
import os
import numpy as np
import torch
from operation_test.quat import *
#from operation_test.quat import *
def read_quat_data_csv(csv_path):
    ori = pd.read_csv(csv_path)
        # print("check_path")
        # print(ori)
    timestamp=pd.DataFrame(ori,columns=["#timestamp"],dtype=np.float64)
    quat=pd.DataFrame(ori,columns=[" q_RS_w []"," q_RS_x []"," q_RS_y []"," q_RS_z []"],dtype=np.float64)
    quat_w=pd.DataFrame(ori,columns=[" q_RS_w []"],dtype=np.float64)
    quat_x=pd.DataFrame(ori,columns=[" q_RS_x []"],dtype=np.float64)
    quat_y=pd.DataFrame(ori,columns=[" q_RS_y []"],dtype=np.float64)
    quat_z = pd.DataFrame(ori, columns=[" q_RS_z []"],dtype=np.float64)
        ###此处数组化
    timestamp=np.array(timestamp.values)
    quat = np.array(quat.values)
    return timestamp,quat
def read_delta_quat_txt(txt_path):
    result=[]
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line=line.strip().split(' ')
            nums = list(map(float, line))
            #print(nums)
            result.append(nums)
        result=np.array(result)
    f.close()
    return result


def read_imu_data_csv(csv_path):
    ori = pd.read_csv(csv_path)
        # print("check_path")
        # print(ori)
    timestamp=pd.DataFrame(ori,columns=["#timestamp"],dtype=np.float64)
    w=pd.DataFrame(ori,columns=["w_RS_S_x [rad s^-1]","w_RS_S_y [rad s^-1]","w_RS_S_z [rad s^-1]"],dtype=np.float64)
    a=pd.DataFrame(ori,columns=["a_RS_S_x [m s^-2]","a_RS_S_y [m s^-2]","a_RS_S_z [m s^-2]"],dtype=np.float64)

   ###此处数组化
    timestamp=np.array(timestamp.values)
    w = np.array(w.values)
    a = np.array(a.values)
    return timestamp,w,a

def read_imu_data_txt(txt_path):
    result=[]
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            nums = list(map(float, line))
            #print(nums)
            result.append(nums)
        result=np.array(result)
        # print(result.shape)
        # print(result)
        f.close()
    t=result[:,0]
    w=result[:,1:4]
    a=result[:,4:7]

    return t,w,a


def get_delta_q_from_data(quat,d_f):
    delta_frame=d_f ###每相邻x帧取　delta_q 此处设为１
    length=quat.shape[0]
    q1=torch.from_numpy(quat[0:length-delta_frame,:])
    q2=torch.from_numpy(quat[delta_frame:,:])
    # print(q1.shape)
    # print(q1)
    # print(q2.shape)
    # print(q2)
    q2=q_inverse(q2)
    delta=qmul(q2,q1)
    # print(delta)
    #delta=torch.exp(delta, out=None)
    #print(delta)
    #delta=torch.log(delta, out=None)
    #print(delta)

    print(delta.shape)
    print(delta)
    return delta

def write_delta_q_data(quat,path):#####传入numpy数据
    if torch.is_tensor(quat):
        print("该数据类型为tensor")
        quat=quat.squeeze(1).numpy()
    print("保存至txt")
    #print(quat)
    np.savetxt(path, quat, fmt='%.8f', delimiter=" ")




    pass
if __name__ == '__main__':
    gt_path="/home/qzd/IMU/informer_op/test_dataset/gt_data.csv"
    imu_path="/home/qzd/IMU/informer_op/test_dataset/imu_data.csv"
    imu_path_txt="/home/qzd/IMU/informer_op/test_dataset/real_imu.txt"
    target_path='/home/qzd/IMU/informer_op/test_dataset/delta_q_data.txt'
    d_frame=50
    gt_times,quat=read_quat_data_csv(gt_path)##从gt中读取q的真值
    imu_times,w,a=read_imu_data_txt(imu_path_txt)####从imu原始数据中读取量测值
    print("check_quat.shape")
    print(quat.shape)
    print("check_w.shape")
    print(w.shape)
    print("check_a.shape")
    print(a.shape)
    delta_q=get_delta_q_from_data(quat,d_frame)
    write_delta_q_data(delta_q,target_path)
    # read_imu_data_txt(imu_path_txt)


