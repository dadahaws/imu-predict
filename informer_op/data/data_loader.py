from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler
from utils.useful_func import *
class quat_estimation_informer(Dataset):
    def __init__(self, gt_data_path,imu_data_path, flag='train'
                 , freq=None,size=None ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 100
            self.label_len = 50
            self.pred_len = 100
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

        self.gt_data_path= gt_data_path
        self.imu_data_path= imu_data_path
        self.__read_data__()

    def __read_data__(self):
        delta_quat=read_delta_quat_txt(self.gt_data_path)
        t, w, a = read_imu_data_txt(self.imu_data_path)
        # print("check_delta_quat.shape")
        # print(delta_quat.shape)
        # print("check_w.shape")
        # print(w.shape)

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
        # print("check_self.times.shape")
        # print(self.times.shape)
        # print("check_feature.shape")
        # print(self.features.shape)
        # print("check_target.shape")
        # print(self.targets.shape)

    def __getitem__(self, index):
        s_begin = index
        # print("check_s_begin")
        # print(s_begin)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len##informer 此处使用　不知道是什么意思
        r_end = r_begin + self.label_len
        # print(s_end-s_begin)
        # print("check_r_index")
        # print(r_end-r_begin)

        time = self.times[s_begin:s_end]
        seq_w = self.features[s_begin:s_end]
        delta_q=self.targets[r_begin:r_end]
        # print("check_getitem.shape")
        # print(seq_quat.shape, time.shape)
        return time,  seq_w ,delta_q

    def __len__(self):
        # print("检查索引上限")####所有样本数量
        # print(len(self.features) - self.seq_len )
        return len(self.features) - self.seq_len
