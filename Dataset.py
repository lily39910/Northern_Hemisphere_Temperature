import numpy as np
import pandas as pd
import torch

class TimeSeriesDataset(torch.utils.data.Dataset):
  def __init__(self, ts:np.array, lookback_size:int, forecast_size:int):
    self.lookback_size = lookback_size
    self.forecast_size = forecast_size
    self.data = ts

  def __len__(self):
    return len(self.data) - self.lookback_size - self.forecast_size + 1

  def __getitem__(self, i):
    idx = (i+self.lookback_size)
    look_back = self.data[i:idx]
    forecast = self.data[idx:idx+self.forecast_size]

    return look_back, forecast
  

class PatchTSDataset(torch.utils.data.Dataset):
  def __init__(self, ts:np.array, patch_size:int=4, n_token:int=6, n_patch:int=4) :
    self.patch_size = patch_size
    self.n_patch = n_patch
    self.n_token = n_token
    self.window_size = int(patch_size * n_patch * n_token / 2)
    self.forecast_size = patch_size
    self.data = ts

  def __len__(self):
    return len(self.data) - self.window_size - self.forecast_size + 1

  def __getitem__(self, i):
    look_back = self.data[i:(i+self.window_size)]
    look_back = np.concatenate([look_back] + [look_back[-self.patch_size:]] * int(self.n_patch / 2))
    x = np.array([look_back[i*int(self.patch_size*self.n_patch/2):(i+2)*int(self.patch_size*self.n_patch/2)] for i in range(self.n_token)])

    y = self.data[(i+self.window_size):(i+self.window_size+self.forecast_size)]
    return x, y
  

class RnnLstmTimeSeriesDataset(torch.utils.data.Dataset):
  def __init__(self, ts:np.array, lookback_size:int, shift_size:int):
    self.lookback_size = lookback_size
    self.shift_size = shift_size
    self.data = ts

  def __len__(self):
    return len(self.data) - self.lookback_size - self.shift_size + 1

  def __getitem__(self, i):
    idx = (i+self.lookback_size)
    look_back = self.data[i:idx]
    forecast = self.data[i+self.shift_size:idx+self.shift_size]

    return look_back, forecast