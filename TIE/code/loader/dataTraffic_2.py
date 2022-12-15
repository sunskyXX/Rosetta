import torchvision
import random
import pandas as pd
import numpy as np
from typing import Any, Callable, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
import os
from scipy.stats import fisk, expon

# class TranformUnique:
#     def __init__(self, )


class DataTraffic:
    def __init__(self, train_path, lossRate_max = 0.4, max_RTT = 0.01, nagle = 'close', dataset = 'train', MSS = 1448):
        self.lossRate_max = lossRate_max
        self.max_RTT = max_RTT
        self.nagle = nagle
        self.MSS = MSS

        self._encoder = {
            'label':    LabelEncoder()
        }
        train_path = os.path.join(os.getcwd(), train_path)
        data = pd.read_csv(train_path)
        features = np.array(data.drop('label', axis=1))
        self.data = np.array(features)
        # self.aug_data = self.__augmentation(self.data)
        # if dataset == 'train':
        #     self.data = self.__augmentation(self.data)
        # self.data = self.__encode_data_X(self.data)
        # self.aug_data = self.__encode_data_X(self.aug_data)
        self.label = np.array(data['label'])
        self.len = len(self.label)
        
        self.mode = dataset
        self.label = self.__encode_data_y(self.label)

    def __augmentation(self, data):
        data1 = data.tolist()
        if self.nagle == 'open':
            data1 = self.train_transform1(data1, self.MSS)
        data1 = self.train_transform2(data1)
        return np.array(data1)

    def __encode_data_X(self, data_X):
        data_X = np.pad(data_X, (0, 300 - len(data_X)), 'constant').reshape(-1, 10, 10)
        data_X = torch.from_numpy(data_X)
        data_X = data_X.float()
        return data_X

    def __encode_data_y(self, data_y):
        self._encoder['label'].fit(list(set(data_y)))
        data_y = self._encoder['label'].transform(data_y)
        return torch.from_numpy(data_y)

    def get_rdelay(self, size):
        delays = []
        while len(delays) < size:
            t = random.random()
            if t < 0.1:
                delays.append(0.21)
            else:
                loc, scale = 7e-06, 0.01094557476340694
                t = expon.rvs(loc = loc, scale = scale, size = 1)
                delays.extend(t)
        return delays
    
    def train_transform1(self, input, MSS):
        res = []
        res.append(input.pop(0))
        res.append(input.pop(0))
        delays = self.get_rdelay(len(input))
        while len(input) > 0:
            RTT = random.random() * self.max_RTT
        # RTT = max_RTT
            buf = 0
            while len(input) > 0 and RTT > 0:
                delay = delays.pop(0)
                RTT -= delay
                buf += int(input.pop(0))
                while buf > MSS:
                    buf -= MSS
                    res.append(MSS)
            res.append(buf)
        res = self.fit_data(res, 100)
        # res = np.array(res)
        return res


    def fit_data(self, seq, tl):
        label = seq[-1]
        data = seq[:-1]
        if len(data) < tl:
            while len(data) < tl:
                data.append(0)
        else:
            data = seq[:tl]
        data.append(label)
        return data

    def train_transform2(self, input):
        res = []
        # print(input.pop(0))
        res.append(input.pop(0))
        lossRate = random.random() * self.lossRate_max
        while len(input) > 0:
            for i in range(len(input)):
                if random.random() > lossRate:
                    res.append(input.pop(i))
                    # print(input.pop(i))
                    break
        # res = np.array(res)
        return res
    


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.data[index]
        aug_data = self.__augmentation(data)
        if self.mode == 'train':
            data = self.__augmentation(data)
        data = self.__encode_data_X(data)
        aug_data = self.__encode_data_X(aug_data)
        target = self.label[index]

        return (data, aug_data), target

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    train_dataset = DataTraffic('train.csv', 0.3, nagle='open')
    train_dataset.data = train_dataset.data[0:2]
    print(train_dataset[1])