from torch.utils.data import Dataset, DataLoader


class quat_estimation_informer(Dataset):
    def __init__(self, root_path,data_path, flag='train'
                 , freq=None,size=None ):
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
