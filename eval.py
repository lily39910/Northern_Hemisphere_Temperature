from Dataset import *
from Metric import *
from Model import *
from Preprocess import *
from train import *
from tqdm.auto import trange
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

def short_term(singleann, single_tst_dl, device, anndataset, tst_size):
  singleann.eval()
  with torch.inference_mode():
    x, y = next(iter(single_tst_dl))
    x, y = x.to(device), y.to(device)
    p = singleann(x)

  scaler = anndataset.get_scaler()
  y = scaler.inverse_transform(y.cpu())
  p = scaler.inverse_transform(p.cpu())

  y = np.concatenate([y[:,0], y[-1,1:]])
  p = np.concatenate([p[:,0], p[-1,1:]])

  plt.title(f"Neural Network, MAPE:{mape(p,y):.4f}, MAE:{mae(p,y):.4f}, R2:{r2_score(p,y):.4f}")
  plt.plot(range(tst_size), y, label="True")
  plt.plot(range(tst_size), p, label="Prediction")
  plt.legend()
  plt.show()

def long_term(data, singleann, single_lookback_size, single_forcast_size, device,single_trn_ds,tst_size,scaler):
  # 장기예측
  window_size = single_lookback_size
  prediction_size = single_forcast_size
  preds = []
  tst_data  = data [-tst_size :]

  #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  singleann.to(device)

  # print(type(x), type(y))
  x, y = single_trn_ds[len(single_trn_ds)-1]

  for _ in range(tst_size):
    y=y.squeeze()
    x = np.concatenate([x,y])[-window_size:]
      
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y=singleann(x_tensor)
    y = y.detach().cpu().numpy()
    y= y.reshape(1,prediction_size)
    preds.append(y)

  preds = np.concatenate(preds, axis=0)
  preds = scaler.inverse_transform(preds).squeeze()
  preds = preds[:,0]

  mape2 = mape(preds,data.Temperature[-tst_size:].to_numpy())
  print("MAPE: ",mape2)
  mae2 = mae(preds,data.Temperature[-tst_size:].to_numpy())
  print("MAE: ",mae2)
  r2_score2 = r2_score(preds,data.Temperature[-tst_size:].to_numpy())
  print("R2_Score: ", r2_score2)

  # 장기예측 plot
  df = pd.DataFrame({"ANN_long": preds}, index=tst_data.index)
  trn, tst = data.Temperature[:-tst_size], data.Temperature[-tst_size:]
  ax = tst.plot(label="TRUE")
  df.plot(ax=ax)
