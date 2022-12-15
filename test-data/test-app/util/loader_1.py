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

# class TranformUnique:
#     def __init__(self, )


class DataTraffic:
    def __init__(self, train_path, lossRate_max = 0.4, MaxAttachRatio = 0.6, nagle = 'close', dataset = 'train'):
        self.lossRate_max = lossRate_max
        self.MaxAttachRatio = MaxAttachRatio
        self.nagle = nagle

        self._encoder = {
            'label':    LabelEncoder()
        }
        train_path = os.path.join(os.getcwd(), train_path)
        data = pd.read_csv(train_path)
        features = np.array(data.drop('label', axis=1))
        self.data = np.array(features)
        self.aug_data = self.__augmentation(self.data)
        if dataset == 'train':
            self.data = self.__augmentation(self.data)
        self.data = self.__encode_data_X(self.data)
        self.aug_data = self.__encode_data_X(self.aug_data)
        self.label = np.array(data['label'])
        self.len = len(self.label)
        
        self.label = self.__encode_data_y(self.label)

    def __augmentation(self, data):
        data1 = data.tolist()
        for i in range(len(data1)):
            if self.nagle == 'open':
                data1[i] = self.train_transform1(data1[i])
            data1[i] = self.train_transform2(data1[i])
        return np.array(data1)

    def __encode_data_X(self, data_X):
        data_X = np.pad(data_X, ((0, 0), (0, 300 - len(data_X[0]))), 'constant').reshape(-1, 3, 10, 10)
        data_X = torch.from_numpy(data_X)
        data_X = data_X.float()
        return data_X

    def __encode_data_y(self, data_y):
        self._encoder['label'].fit(list(set(data_y)))
        data_y = self._encoder['label'].transform(data_y)
        return torch.from_numpy(data_y)

    def train_transform1(self, input):
        ratio = self.MaxAttachRatio * random.random()
        res = []
        res.append(input.pop(0))
        res.append(input.pop(0))
        while len(input) > 0:
            if random.random() > ratio:
                res.append(input.pop(0))
            else:
                t = res[-1] + input.pop(0)
                res[-1] = t
                while t > 1460:
                    res[-1] = 1460
                    t -= 1460
                res.append(t)
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
    
    # def train_transform3():




    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        train, aug_train, target = self.data[index], self.aug_data[index], self.label[index]
        return (train, aug_train), target

    def __len__(self):
        return self.len

if __name__ == '__main__':
    train_dataset = DataTraffic('train.csv', 0.4, nagle='open', dataset = 'train')
    print(train_dataset[6])