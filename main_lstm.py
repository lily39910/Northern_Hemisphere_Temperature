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
  import matplotlib.pyplot as plt
  print(f'사용 모델은 : {args.model}')

  train_params = cfg.get('train_params')
  device = train_params.get('device')
  
  files = cfg.get('files')
  data = pd.read_csv(files.get('data'))
  data['YearMonth'] = pd.to_datetime(data['YearMonth']) 

  Optim = train_params.get('optim')
  optim_params = train_params.get('optim_params')
  epochs = train_params.get('epochs')

  if 'Multi' == args.model:
    data['rolling_mean'] = data.rolling(12).mean()
    data['diff1'] = data['Temperature'].diff(1)
    data = data.dropna()
  data.set_index('YearMonth', inplace=True)

  dl_params = train_params.get('data_loader_params')
  tst_size = dl_params.get('tst_size')
  trn, tst = data[:-tst_size], data[-tst_size:]
  if 'Stateful' == args.model:
    print('this is stateful')
    stateful = cfg.get(args.model)
    statefulmodel = stateful.get('model')
    stateful_params = stateful.get('model_params')

    # dataset params
    statefuldata = StatefulLoadData(data,tst_size)
    trn_scaled, tst_scaled = statefuldata.scaling()

    # model params
    state_input_size = stateful_params.get('state_input_size')     #고정
    state_hidden_size = stateful_params.get('state_hidden_size')
    state_output_size = stateful_params.get('state_output_size')      #고정
    state_num_layers = stateful_params.get('state_num_layers')
    state_batch_size = stateful_params.get('state_batch_size')

    statefulmodel = statefulmodel(state_input_size,state_hidden_size,state_output_size,state_num_layers)
    statefulmodel.to(device)

    trn_x = torch.tensor(trn_scaled[:-1]).split(state_batch_size)
    trn_y = torch.tensor(trn_scaled[1:]).split(state_batch_size)
    tst_y = torch.tensor(tst_scaled)

    optim = Optim(statefulmodel.parameters(), optim_params.get('lr'))

    p = state_train_epoch(statefulmodel, epochs , device, zip(trn_x,trn_y), optim, tst_y)

    scaler = statefuldata.get_scaler()
    prd = scaler.inverse_transform(p.cpu()[:,:1])
    torch.save(statefulmodel, f'./modelpth/Stateful/{args.model}_{optim_params.get("name")}_epochs_{epochs}_hidden_size_{state_hidden_size}_batch_size_{state_batch_size}.pth')
    plt.title(f"LSTM (Stateful), MAPE:{mape(prd,tst.to_numpy()):.4f}, MAE:{mae(prd,tst.to_numpy()):.4f}, R2:{r2_score(prd,tst.to_numpy())}")
    plt.plot(tst.Temperature.to_numpy(), label='TST')
    plt.plot(prd, label='PRD')
    plt.legend()
    plt.show()

  elif 'Stateless' == args.model:
    print('this is stateless')
    stateless = cfg.get(args.model)
    statelessmodel = stateless.get('model')
    stateless_params = stateless.get('model_params')
    #Multi Load Data
    stateless_window_size = stateless_params.get('stateless_window_size')
  
    # Multi model params
    stateless_input_size = stateless_params.get('stateless_input_size')
    stateless_hidden_size = stateless_params.get('stateless_hidden_size')
    stateless_output_size = stateless_params.get('stateless_output_size')
    stateless_num_layers = stateless_params.get('stateless_num_layers')

    statelessdata = StatelessLoadData(data, tst_size, stateless_window_size)
    stateless_trn_dl , stateless_tst_dl = statelessdata.scaling()
    tst_y = statelessdata.get_scaled()
    
    statelesslstm = statelessmodel(stateless_input_size , stateless_hidden_size , stateless_output_size , stateless_num_layers)
    statelesslstm.to(device)

    optim = Optim(statelesslstm.parameters(), optim_params.get('lr'))

    p = stateless_train_epoch(statelesslstm,epochs,device,stateless_trn_dl,optim,stateless_tst_dl,tst_y)
    print(p)
    scaler = statelessdata.get_scaler()
    prd = scaler.inverse_transform(p.cpu())
    # prd = p.cpu()
    
    plt.title(f"LSTM (Look-back window), MAPE:{mape(prd,tst.to_numpy()):.4f}, MAE:{mae(prd,tst.to_numpy()):.4f}, R2:{r2_score(prd,tst.to_numpy())}")
    plt.plot(tst.to_numpy()[:,:1], label='TST')
    plt.plot(prd, label='PRD')
    plt.legend()
    plt.show()

  flag = input(f'{args.model}_{optim_params.get("name")}_tst_size_{tst_size}_epochs_{epochs}_lr_{optim_params.get("lr")}_hidden_size_{stateless_hidden_size}_window_size{stateless_window_size}.pth 저장하시겠습니까? 1 or 0')
  if flag == '1':
    torch.save(statelesslstm, f'./modelpth/Stateless/{args.model}_tst_size_{tst_size}_{optim_params.get("name")}_epochs_{epochs}_lr_{optim_params.get("lr")}_hidden_size_{stateless_hidden_size}_window_size{stateless_window_size}.pth')
    
  print('finish')
  



def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")
  parser.add_argument("-m", "--model", default="Stateful", type=str, help="What models do you learn?")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  main(config, args)