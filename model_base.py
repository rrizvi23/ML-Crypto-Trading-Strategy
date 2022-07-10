import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.utils.data.dataloader as dl
import dataset
import scipy.stats as st

print("Reading data...")
info = pd.read_csv('asset_details.csv')
train = pd.read_csv('train.csv')
#train = train.loc[train["Asset_ID"]==0]
#Columns = Timestamp, Asset_ID, Count, Open, High, Low, Close, Volume, VWAP, Target
print("Finished.")

def ResidualizeMarket(df, mktColumn, window):
    if mktColumn not in df.columns:
        return df

    mkt = df[mktColumn]

    num = df.multiply(mkt.values, axis=0).rolling(window).mean().values  #numerator of linear regression coefficient
    denom = mkt.multiply(mkt.values, axis=0).rolling(window).mean().values  #denominator of linear regression coefficient
    beta = np.nan_to_num( num.T / denom, nan=0., posinf=0., neginf=0.)  #if regression fell over, use beta of 0

    resultRet = df - (beta * mkt.values).T  #perform residualization
    resultBeta = 0.*df + beta.T  #shape beta

    return resultRet.drop(columns=[mktColumn]), resultBeta.drop(columns=[mktColumn]) 
        
def get_target(close, info):
    ids = list(info.Asset_ID)
    names = list(info.Asset_Name)
    targets = pd.DataFrame(index=close.index)
    
    for i, id in enumerate(ids):
        asset = close[id]
        targets[names[i]] = (
            asset.shift(periods=-16) /
            asset.shift(periods=-1)
        ) - 1
    
    weights = np.array(list(info.Weight))
    targets['m'] = np.average(targets.fillna(0), axis=1, weights=weights)
    m = targets['m']

    numer = targets.multiply(m.values, axis=0).rolling(3750).mean().values
    denom = m.multiply(m.values, axis=0).rolling(3750).mean().values
    beta = np.nan_to_num(numer.T / denom, nan=0, posinf=0, neginf=0)
    targets = targets - (beta * m.values).T
    return targets


print("Treating data...")
close = train.pivot(index=["timestamp"], columns=["Asset_ID"], values=["Close"])["Close"]
close.index = pd.to_datetime(close.index, unit='s')
close.interpolate(method='time')
target = get_target(close, info)
target = target.drop(columns=['m'])
print("Finished.")

## Taking a single asset for training speed.
close = close[1]
target = target["Bitcoin"]

boundary_train = int(0.8*len(close))
target1 = torch.tensor(target.iloc[:boundary_train])
target2 = torch.tensor(target.iloc[boundary_train:])
close1 = torch.tensor(close.iloc[:boundary_train])
close2 = torch.tensor(close.iloc[boundary_train:])
mu = close1[~close1.isnan()].mean()
std = close1[~close1.isnan()].std()
close1 = (close1 - mu) / std
close2 = (close2 - mu) / std

NUM_EPOCHS = 200
INP_DIM = 60
BATCH_SIZE = 128

ds1 = dataset.ClumpedDataset(close1, target1, INP_DIM)
ds2 = dataset.ClumpedDataset(close2, target2, INP_DIM)
train_ds = dl.DataLoader(ds1, BATCH_SIZE, True)
test_ds = dl.DataLoader(ds2, BATCH_SIZE, True)

def run_model(model, optim, loss_f):
    model.train()
    model.double()
    losses = []
    print("Starting training...")

    for e in range(NUM_EPOCHS):
        optim.zero_grad()

        for x, y in train_ds:
            if torch.sum(torch.isnan(x)) > 0 or torch.sum(torch.isnan(y)) > 0:
                continue
            output = model.forward(x)
            loss = loss_f(output, y)

            losses.append(loss)
            loss.backward()
            optim.step()

        if e % 10 == 0:
            print("Epoch " + str(e))
            #torch.save(model.state_dict(), 'checkpoint'+str(e)+'.pt')
        
    print("Finished training.")

    plt.figure()
    plt.plot(losses)
    plt.yscale('log')
    plt.savefig('train.png')    

def view_hist(model):
    true = []
    pred = []
    model.double()
    tot = 0
    epsilon = [0, 1e-4, 1.5e-4, 2e-4, 2.5e-4]
    acc = [0 for i in range(len(epsilon))]
    for x, y in test_ds:
        if torch.sum(torch.isnan(x)) > 0 or torch.sum(torch.isnan(y)) > 0:
                continue
        op = model.forward(x)
        true.extend(y[:, -1].tolist())
        pred.extend(op[:, -1].tolist())

        y = y[:, -1]
        op = op[:, -1]        
        for j in range(len(epsilon)):
            eps = epsilon[j]
            y = y[(op > eps) | (op < -eps)]
            op = op[(op > eps) | (op < -eps)]

            acc[j] += torch.sum((y > 0) == (op > 0))
        tot += len(y)
    acc = [a / tot for a in acc]
    plt.figure()
    plt.hist(true, label='true')    
    plt.hist(pred, label='pred')
    print(acc)
    plt.savefig('hist.png')