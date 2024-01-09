import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from Dataset import *

#Arima는 scaling X
#Patch는 다른거.

#NN에서만?
#그럼 diff,rolling은 csv를 따로 만들어서 그냥 불러다 쓰고 class를 적용한다?
class ANNLoadData():
    def __init__(self, data:pd.DataFrame, lookback_size :int, forecast_size:int, tst_size:int, batch_size:int):
        self.lookback_size = lookback_size
        self.forecast_size = forecast_size
        self.tst_size = tst_size
        self.data = data
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()

    def scaling(self):
        
        trn_scaled = self.scaler.fit_transform(self.data[:-self.tst_size].to_numpy(dtype=np.float32)).flatten()
        tst_scaled = self.scaler.transform(self.data[-self.tst_size-self.lookback_size:].to_numpy(dtype=np.float32)).flatten()

        trn_ds = TimeSeriesDataset(trn_scaled, self.lookback_size, self.forecast_size)
        tst_ds = TimeSeriesDataset(tst_scaled, self.lookback_size, self.forecast_size)

        trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=self.batch_size, shuffle=True)
        tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=self.tst_size, shuffle=False)
        return trn_dl, tst_dl , trn_ds
    
    def get_scaler(self):
        return self.scaler
    

class PatchLoadData():
    def __init__(self, data:pd.DataFrame, patch_size:int, n_patch:int, n_token:int, tst_size:int , batch_size:int):
        self.tst_size = tst_size
        self.data = data
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_patch = n_patch
        self.n_token = n_token
        self.scaler = MinMaxScaler()

    def scaling(self):
        window_size = int(self.patch_size * self.n_patch * self.n_token / 2)

        
        trn_scaled = self.scaler.fit_transform(self.data[:-self.tst_size].to_numpy(dtype=np.float32)).flatten()
        tst_scaled = self.scaler.transform(self.data[-self.tst_size-window_size:].to_numpy(dtype=np.float32)).flatten()

        trn_ds = PatchTSDataset(trn_scaled, self.patch_size, self.n_token, self.n_patch) #4, 6
        tst_ds = PatchTSDataset(tst_scaled, self.patch_size, self.n_token, self.n_patch) #4, 6

        trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=self.batch_size, shuffle=True)
        tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=self.tst_size, shuffle=False)
        return trn_dl, tst_dl, trn_ds
    

    def get_scaler(self):
        return self.scaler


class StatefulLoadData():
    def __init__(self, data:pd.DataFrame ,tst_size:int):
        self.tst_size = tst_size
        self.data = data
        self.scaler = MinMaxScaler()
        self.scaler_ra = MinMaxScaler()


    def scaling(self):
        tst_size = self.tst_size

        #data_mw = self.data.copy()
        self.data['rolling_avg'] = self.data.Temperature.rolling(12).mean()
        self.data = self.data.dropna()

        trn, tst = self.data[:-tst_size], self.data[-tst_size:]
        trn_scaled, tst_scaled = trn.copy(), tst.copy()

        trn_scaled['Temperature'] = self.scaler.fit_transform(trn.Temperature.to_numpy(np.float32).reshape(-1,1))
        trn_scaled['rolling_avg'] = self.scaler_ra.fit_transform(trn.rolling_avg.to_numpy(np.float32).reshape(-1,1))

        tst_scaled['Temperature'] = self.scaler.transform(tst.Temperature.to_numpy(np.float32).reshape(-1,1))
        tst_scaled['rolling_avg'] = self.scaler_ra.transform(tst.rolling_avg.to_numpy(np.float32).reshape(-1,1))

        trn_scaled = trn_scaled.to_numpy(np.float32)
        tst_scaled = tst_scaled.to_numpy(np.float32)
        return trn_scaled, tst_scaled
    
    def get_scaler(self):
        return self.scaler

    def get_scaler_ra(self):
        return self.scaler_ra
    
class StatelessLoadData():
    def __init__(self, data:pd.DataFrame, tst_size:int, window_size:int):
        self.tst_size = tst_size
        self.data = data
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.scaler_ra = MinMaxScaler()

    def scaling(self):
        tst_size = self.tst_size
        trn, tst = self.data[:-tst_size], self.data[-tst_size:]

        

        trn_scaled, tst_scaled = trn.copy(), tst.copy()

        trn_scaled['Temperature'] = self.scaler.fit_transform(trn.Temperature.to_numpy(np.float32).reshape(-1,1))
        trn_scaled['rolling_avg'] = self.scaler_ra.fit_transform(trn.rolling_avg.to_numpy(np.float32).reshape(-1,1))

        tst_scaled['Temperature'] = self.scaler.transform(tst.Temperature.to_numpy(np.float32).reshape(-1,1))
        tst_scaled['rolling_avg'] = self.scaler_ra.transform(tst.rolling_avg.to_numpy(np.float32).reshape(-1,1))

        tst_scaled = tst_scaled.dropna()
        trn_scaled = trn_scaled.dropna()

        trn_scaled = trn_scaled.to_numpy(np.float32)
        tst_scaled = tst_scaled.to_numpy(np.float32)
        #print(trn_scaled.shape, tst_scaled.shape)

        trn_ds = RnnLstmTimeSeriesDataset(trn_scaled, self.window_size, 1)
        tst_ds = RnnLstmTimeSeriesDataset(np.concatenate([trn_scaled[-self.window_size:], tst_scaled], axis=0), self.window_size, 1)
        #print(trn_ds.shape, tst_ds.shape)

        trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=64, shuffle=True)
        tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=len(tst_ds), shuffle=False)
        return trn_dl, tst_dl
    
    def get_scaler(self):
        return self.scaler

    def get_scaler_ra(self):
        return self.scaler_ra
    
class MultiANNLoadData():
    def __init__(self, data:pd.DataFrame, lookback_size :int, forecast_size:int, tst_size:int, batch_size:int):
        self.lookback_size = lookback_size
        self.forecast_size = forecast_size
        self.tst_size = tst_size
        self.data = data
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()

    def scaling(self):
        
        trn_scaled = self.scaler.fit_transform(self.data[:-self.tst_size].to_numpy(dtype=np.float32))
        tst_scaled = self.scaler.transform(self.data[-self.tst_size-self.lookback_size:].to_numpy(dtype=np.float32))

        trn_ds = TimeSeriesDataset(trn_scaled, self.lookback_size, self.forecast_size)
        tst_ds = TimeSeriesDataset(tst_scaled, self.lookback_size, self.forecast_size)

        trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=self.batch_size, shuffle=True)
        tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=self.tst_size, shuffle=False)
        return trn_dl, tst_dl , trn_ds
    
    def get_scaler(self):
        return self.scaler