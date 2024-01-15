import torch
import torch.nn.functional as F
import torchmetrics
from Model import *

config = {
    'files':{
        'data': '../data/final.csv',
    },

    'ANN' : {
        'model' : Net,
        'model_params' : {
            'single_lookback_size':96,
            'single_forcast_size':12,
            'single_batch_size':32,
            'single_d_in' : 96,
            'single_d_out' : 12,
            'single_hidden_dim' : 512,
            'activation' : F.relu,
        },
    },
    'Multi' : {
        'model' : NetMulti,
        'model_params' : {
            'multi_lookback_size': 80,
            'multi_forcast_size':20,
            'multi_batch_size':64,
            'multi_hidden_size' : 512,
            'multi_channel_size': 4,
            'activation' : F.relu,
        },
    },
    'PatchTST' : {
        'model' : PatchTST,
        'model_params' : {
            'patch_length' : 16, # fix
            'n_patch' : 4,
            'n_token' : 64, # what?
            'patch_batch_size' : 32,
            'patch_model_dim' : 128, # fix
            'patch_num_heads' : 16, # fix
            'patch_num_layers' : 3, # fix
            'patch_output_dim' : 20, # fix 
        },
    },
    'Stateful' : {
        'model' : StatefulLSTM,
        'model_params' : {
            'state_input_size' : 2,     # fix
            'state_hidden_size' : 32,
            'state_output_size' : 2,     # fix
            'state_num_layers' : 2,
            'state_batch_size' : 512,
        },
    },
    'Stateless' : {
        'model' : StatelessLSTM,
        'model_params' : {
            'stateless_window_size' : 64,
            'stateless_input_size' : 2,
            'stateless_hidden_size' : 64,
            'stateless_output_size' : 2,
            'stateless_num_layers' : 2,
        },
    },

    'train_params': {
        'data_loader_params' : {
            'batch_size' : 32,
            'suffle' : True,
            'tst_size' : 20,
        },
        'loss' : F.mse_loss,
        'optim' : torch.optim.AdamW,
        'optim_params' : {
            'lr' : 0.0001,
            'name':'AdamW',
        },
        'metric' : '',
        'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'epochs' : 50, # Multi 300
  }
}