import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
  def __init__(self, d_in, d_out, d_hidden, activation=F.relu):
    super().__init__()
    self.lin1 = nn.Linear(d_in, d_hidden)
    self.lin2 = nn.Linear(d_hidden, d_out)
    self.activation = activation

  def forward(self, x):
    x = self.lin1(x)
    x = self.activation(x)
    x = self.lin2(x)
    return F.sigmoid(x)
  



class NetMulti(nn.Module):
  def __init__(self, d_in, d_out, d_hidden, c_in, activation=F.relu):
    super().__init__()
    self.lin1 = nn.Linear(d_in*c_in, d_hidden)
    self.lin2 = nn.Linear(d_hidden, d_hidden)
    self.lin3 = nn.Linear(d_hidden, d_out*c_in)
    self.activation = activation
    self.c_in = c_in
    self.d_out = d_out

  def forward(self, x):
    x = x.flatten(1)    # (B, d_in * c_in)
    x = self.lin1(x)    # (B, d_hidden)
    x = self.activation(x)
    for _ in range(2):
      x = self.lin2(x)    # (d_hidden, d_hidden)
      x = self.activation(x)
    x = self.lin3(x).reshape(-1, self.d_out, self.c_in)    # (B, d_out, c_in)
    return F.sigmoid(x)
  



class PatchTST(nn.Module):
  def __init__(self, n_token, input_dim, model_dim, num_heads, num_layers, output_dim):
    super(PatchTST, self).__init__()
    self.patch_embedding = nn.Linear(input_dim, model_dim)    # Input Embedding
    self._pos = torch.nn.Parameter(torch.randn(1,1,model_dim))  # Positional Embedding

    encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    self.output_layer = nn.Linear(model_dim * n_token, output_dim)

  def forward(self, x):
    # x shape: (batch_size, n_token, token_size)
    x = self.patch_embedding(x)   # (batch_size, n_token, model_dim)
    x = x + self._pos
    x = self.transformer_encoder(x)   # (batch_size, n_token, model_dim)
    x = x.view(x.size(0), -1)       # (batch_size, n_token * model_dim)
    output = self.output_layer(x)   # (batch_size, out_dim =4 patch_size == 4)
    return F.sigmoid(output)
  



class StatefulLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super().__init__()
    self.reset_state()
    self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
    self.head = nn.Linear(hidden_size, output_size)

  def reset_state(self, state=None):
    self.state = state

  def forward(self, x):
    assert x.dim() == 2   # (sequence_length, input_size)
    if self.state is None:
      x, (hn, cn) = self.rnn(x)   # state will be set to be zeros by default
    else:
      x, (hn, cn) = self.rnn(x, self.state)   # pass the saved state
    # x.shape == (sequence_length, hidden_size)
    self.state = (hn.detach(), cn.detach())   # save the state
    x = self.head(x)  # (sequence_length, output_size)
    return F.sigmoid(x)

  def predict(self, x0, steps, state=None):
    if state is not None:
      self.reset_state(state)
    output = []
    x = x0.reshape(1,-1)
    for i in range(steps):
      x = self.forward(x)
      output.append(x)
    return torch.concat(output, 0)
  




class StatelessLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super().__init__()
    self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.head = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x, _ = self.rnn(x)   # state will be set to be zeros by default
    # x.shape == (batch_size, sequence_length, hidden_size)
    x = self.head(x)  # (batch_size, sequence_length, output_size)
    return F.sigmoid(x)

  def predict(self, x, steps, state=None):
    output = []
    for i in range(steps):
      x = self.forward(x)
      output.append(x[-1:])
    return torch.concat(output, 0)  