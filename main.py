from Dataset import *
from Metric import *
from Model import *
from Preprocess import *
from train import *
from tqdm.auto import trange
import torch.nn.functional as F

def main(cfg, args):
  import numpy as np
  import pandas as pd
  print(f'사용 모델은 : {args.model}')

  train_params = cfg.get('train_params')
  device = train_params.get('device')
  
  files = cfg.get('files')
  data = pd.read_csv(files.get('data'))
  data['YearMonth'] = pd.to_datetime(data['YearMonth']) 

  if 'Multi' == args.model:
    data['rolling_mean'] = data.rolling(12).mean()
    data['diff1'] = data['Temperature'].diff(1)
    data['rolling_mean4'] = data['Temperature'].rolling(4).mean()
    data = data.dropna()
  data.set_index('YearMonth', inplace=True)

  dl_params = train_params.get('data_loader_params')
  tst_size = dl_params.get('tst_size')
  if 'ANN' == args.model:
    ann = cfg.get(args.model)
    singleann = ann.get('model')
    ann_params = ann.get('model_params')

    # dataset params
    single_lookback_size = ann_params.get('single_lookback_size')
    single_forcast_size = ann_params.get('single_forcast_size')
    single_batch_size = ann_params.get('single_batch_size')
    trn, tst = data[:-tst_size], data[-tst_size:]

    anndataset = ANNLoadData(data,single_lookback_size,single_forcast_size,tst_size,single_batch_size)
    trn_dl , tst_dl, trn_ds = anndataset.scaling()

    # model params
    single_d_in = ann_params.get('single_d_in')
    single_d_out = ann_params.get('single_d_out')
    single_d_hidden = ann_params.get('single_hidden_dim')
    model = singleann(single_d_in,single_d_out,single_d_hidden)
    model.to(device)
  elif 'Multi' == args.model:
    ann = cfg.get(args.model)
    multiann = ann.get('model')
    multiann_params = ann.get('model_params')
    #Multi Load Data
    multi_lookback_size = multiann_params.get('multi_lookback_size')
    multi_forecast_size = multiann_params.get('multi_forcast_size')
    multi_batch_size = multiann_params.get('multi_batch_size')

    multidata = MultiANNLoadData(data,multi_lookback_size,multi_forecast_size,tst_size,multi_batch_size)
    trn_dl, tst_dl , trn_ds = multidata.scaling() 

    # Multi model params
    multi_input_size = multi_lookback_size   #고정
    multi_hidden_size = multiann_params.get('multi_hidden_size')
    multi_output_size = multi_forecast_size     #고정
    multi_channel_size = multiann_params.get('multi_channel_size')

    model = multiann(multi_lookback_size,multi_output_size,multi_hidden_size,multi_channel_size)
    model.to(device)
  elif 'PatchTST' == args.model:
    tst = cfg.get(args.model)
    patchtst = tst.get('model')
    patchtst_params = tst.get('model_params')
    # patch load data
    patch_length = patchtst_params.get('patch_length') #고정
    n_patch = patchtst_params.get('n_patch')
    n_token = patchtst_params.get('n_token')   #조정
    patch_batch_size = patchtst_params.get('patch_batch_size')

    patchdata = PatchLoadData(data, n_patch, patch_length, tst_size, patch_batch_size)
    trn_dl , tst_dl, trn_ds = patchdata.scaling()

    # patch model params
    patch_model_dim = patchtst_params.get('patch_model_dim') #고정
    patch_num_heads = patchtst_params.get('patch_num_heads')   #고정
    patch_num_layers = patchtst_params.get('patch_num_layers')   #고정
    patch_output_dim = patchtst_params.get('patch_output_dim')  #고정

    model = patchtst(n_patch, patch_length, patch_model_dim, patch_num_heads, patch_num_layers, patch_output_dim)
    model.to(device)

  Optim = train_params.get('optim')
  optim_params = train_params.get('optim_params')
  optim = Optim(model.parameters(), optim_params.get('lr'))
  epochs = train_params.get('epochs')

  train_epoch(model, epochs, device, trn_dl, optim, trn_ds, tst_dl)
  if args.model == 'ANN':
    torch.save(model, f'./modelpth/ANN/{args.model}_{optim_params.get("name")}_epochs_{epochs}_lookback_{single_lookback_size}_forecast_{single_forcast_size}.pth')
  elif args.model == 'Multi':
    torch.save(model, f'./modelpth/Multi/{args.model}_{optim_params.get("name")}_epochs_{epochs}_lookback_{multi_lookback_size}_forecast_{multi_forecast_size}_channel_{multi_channel_size}.pth')
  elif args.model == 'PatchTST':
    torch.save(model.state_dict(), f'./modelpth/PatchTST/{args.model}_{optim_params.get("name")}_epochs_{epochs}_n_patch_{n_patch}_patch_length_{patch_length}.pth')



def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")
  parser.add_argument("-m", "--model", default="ANN", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  main(config, args)