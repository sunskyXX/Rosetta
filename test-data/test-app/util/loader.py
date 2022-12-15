import numpy as np
import pandas as pd
# from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class KddData(object):

    def __init__(self, batch_size, file_name2):
        data1 = pd.read_csv(file_name2)
        data2 = pd.read_csv(file_name2)
        
        
        self._encoder = {
            'label':    LabelEncoder()
        }
        self.batch_size = batch_size

        target = np.array(data1['label'])
        features = np.array(data1.drop('label', axis=1))
        data_X, data_y = self.__encode_data(features, target)
        self.train_dataset = TensorDataset(
            torch.from_numpy(data_X.astype(np.float32)),
            torch.from_numpy(data_y.astype(np.int64))
        )

        target = np.array(data2['label'])
        features = np.array(data2.drop('label', axis=1))
        data_X, data_y = self.__encode_data(features, target)
        self.test_dataset = TensorDataset(
            torch.from_numpy(data_X.astype(np.float32)),
            torch.from_numpy(data_y.astype(np.int64))
        )

 
        self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, self.batch_size, shuffle=True)

    """将数据中字符串部分转换为数字，并将输入的41维特征转换为8*8的矩阵"""
    def __encode_data(self, data_X, data_y):
        self._encoder['label'].fit(list(set(data_y)))
        data_X = np.pad(data_X, ((0, 0), (0, 100 - len(data_X[0]))), 'constant').reshape(-1, 1, 10, 10)
        data_y = self._encoder['label'].transform(data_y)
        return data_X, data_y

    """将数据拆分为训练集和测试集，并转换为TensorDataset对象"""
    def __split_data_to_tensor(self, data_X, data_y):
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3)
        train_dataset = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.int64))
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test.astype(np.float32)),
            torch.from_numpy(y_test.astype(np.int64))
        )
        return train_dataset, test_dataset

    """接受一个数组进行解码"""
    def decode(self, data, label=False):
        if not label:
            _data = list(data)
            _data[1] = self._encoder['protocal'].inverse_transform([_data[1]])[0]
            _data[2] = self._encoder['service'].inverse_transform([_data[2]])[0]
            _data[2] = self._encoder['flag'].inverse_transform([_data[3]])[0]
            return _data
        return self._encoder['label'].inverse_transform(data)
    
    def encode(self, data, label=False):
        if not label:
            _data = list(data)
            _data[1] = self._encoder['protocal'].transform([_data[1]])[0]
            _data[2] = self._encoder['service'].transform([_data[2]])[0]
            _data[3] = self._encoder['flag'].transform([_data[3]])[0]
            return _data
        return self._encoder['label'].transform([data])[0]