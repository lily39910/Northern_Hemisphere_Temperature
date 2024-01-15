import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
import torch.nn.functional as F
from tqdm.auto import trange


def state_train_epoch(model, epochs, device, data, state_optim , tst_y):
    #state_optim = torch.optim.AdamW(model.parameters(), lr=lr)
    trn_losses = []  # 훈련 손실을 저장할 리스트
    tst_losses = []  # 테스트 손실을 저장할 리스트

    data = tuple(data)
    pbar = trange(epochs)
    for e in pbar:
        model.train()
        model.reset_state()
        trn_loss = .0
        count = 0
        
        for x, y in data:   #zip(trn_x, trn_y)
            #print(x,y,len(y))
            x, y = x.to(device), y.to(device)
            state_optim.zero_grad()
            p = model(x)
            loss = F.mse_loss(p, y)
            loss.backward()
            state_optim.step()
            count += len(y)
            trn_loss += loss.item()*len(y)
        #print(trn_loss,count)
        trn_loss = trn_loss/count
        trn_losses.append(trn_loss)

        model.eval()
        with torch.inference_mode():
            p = model.predict(y[-1:].to(device), len(tst_y))
            tst_loss = F.mse_loss(p, tst_y.to(device)).item()
            tst_losses.append(tst_loss)

        pbar.set_postfix({'trn_loss': trn_loss, 'tst_loss': tst_loss})

    # 에폭별 손실 그래프
    plt.plot(range(1, epochs + 1), trn_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), tst_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return p


def stateless_train_epoch(model, epochs, device, data, optim , tst_dl, tst_scaled):
    trn_losses = []  # 훈련 손실을 저장할 리스트
    tst_losses = []  # 테스트 손실을 저장할 리스트

    pbar = trange(epochs)
    for e in pbar:
        model.train()
        trn_loss = .0
        count = 0
        for x, y in data:
            #print(x.shape, y.shape)
            #print(y)
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            p = model(x)
            loss = F.mse_loss(p, y)
            loss.backward()
            optim.step()
            count += len(y)
            trn_loss += loss.item()*len(y)
        trn_loss = trn_loss/count
        trn_losses.append(trn_loss)

        model.eval()
        with torch.inference_mode():
            x, y = next(iter(tst_dl))
            p = model.predict(x[0].to(device), len(tst_scaled))[:,:1]
            tst_loss = F.mse_loss(p, torch.tensor(tst_scaled[:,:1]).view(-1,1).to(device)).item()
        tst_losses.append(tst_loss)

        pbar.set_postfix({'trn_loss': trn_loss, 'tst_loss': tst_loss})

    # 에폭별 손실 그래프
    plt.plot(range(1, epochs + 1), trn_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), tst_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return p


def train_epoch(model, epochs, device, data, optim, single_trn_ds, single_tst_dl):
    trn_losses = []  # 훈련 손실을 저장할 리스트
    tst_losses = []  # 테스트 손실을 저장할 리스트
    
    pbar = trange(epochs)
    for i in pbar:
        model.train()
        trn_loss = .0
        for x, y in data:
            x, y = x.to(device), y.to(device)
            p = model(x)
            #print(p)
            optim.zero_grad()
            loss = F.mse_loss(p, y)
            #print(loss)
            loss.backward()
            optim.step()
            trn_loss += loss.item()*len(y)
        trn_loss = trn_loss/len(single_trn_ds)
        trn_losses.append(trn_loss)

        model.eval()
        with torch.inference_mode():
            x, y = next(iter(single_tst_dl))
            x, y = x.to(device), y.to(device)
            p = model(x)
            tst_loss = F.mse_loss(p,y)
            tst_losses.append(tst_loss.item())
            # tst_mape = mape(p,y)
            # tst_mae = mae(p,y)
        pbar.set_postfix({'loss':trn_loss, 'tst_loss':tst_loss.item()})#, 'tst_mape':tst_mape.item(), 'tst_mae':tst_mae.item()})
    
    # 에폭별 손실 그래프
    plt.plot(range(1, epochs + 1), trn_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), tst_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return p