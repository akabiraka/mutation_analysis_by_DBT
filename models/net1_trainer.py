import sys
sys.path.append("../mutation_analysis_by_DBT")

from models.Net1 import Net1
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from data_generators.dataset import get_batched_data

# configurations
train_data_path = "data/datasets/train.csv"
val_data_path = "data/datasets/val.csv"
run_no = 0
lr = 0.0001
n_epochs = 50
batch_size = 64

# dataset
train_batched_df = get_batched_data(train_data_path, batch_size)
val_df = pd.read_csv(val_data_path)

# model and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net1(device).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

# training the model
best_pf, best_loss = 0.0, np.inf
train_losses, val_losses = [], []

def init_model():
    # initializes the model to bestrandom weights 
    model_path = "outputs/saved_models/run_{}_lr_{}_batch_{}.pt".format(run_no, lr, batch_size)
    n_init = 1000
    best_pr = 0.0
    for i in range(1, n_init+1):
        print("training ...")
        train_loss = model.train_loop(train_batched_df, optimizer)
        train_losses.append(train_loss)
        print("validating ...")
        val_loss, trgt_ddg_list, pred_ddg_list = model.validate(val_df)
        val_losses.append(val_loss)
        r, p = stats.pearsonr(trgt_ddg_list, pred_ddg_list)
        print("[{}/{}] train_loss:{:.4f}, val_loss:{:.4f}, Pr:{:.4f}".format(i, n_init, train_loss, val_loss, r))
        if r > best_pr: 
            torch.save(model.state_dict(), model_path) 
            best_pr = r
            print("Best pearson: ", best_pr)
        else:
            model = Net1(device).to(device)

    print("Best pearson: ", best_pr) #0.29

def train_from_init():
    inp_model_path = "outputs/saved_models/init_run_0_lr_1e-05_batch_64.pt".format(run_no, lr, batch_size)
    out_model_path = "outputs/saved_models/from_init_run_{}_lr_{}_batch_{}.pt".format(run_no, lr, batch_size)
    best_pr = 0.30
    model.load_state_dict(torch.load(inp_model_path))
    print(model)
    for epoch in range(1, n_epochs+1):
        train_loss = model.train_loop(train_batched_df, optimizer)
        train_losses.append(train_loss)
        val_loss, trgt_ddg_list, pred_ddg_list = model.validate(val_df)
        val_losses.append(val_loss)
        r, p = stats.pearsonr(trgt_ddg_list, pred_ddg_list)
        print("[{}/{}] train_loss:{:.4f}, val_loss:{:.4f}, Pr:{:.4f}".format(epoch, n_epochs, train_loss, val_loss, r))
        if r > best_pr: 
            torch.save(model.state_dict(), out_model_path) 
            best_pr = r
            print("Best pearson: ", best_pr)

    print("Best pearson: ", best_pr)#this could not improve the initial pr

if __name__=="__main__":
    # init_model()
    train_from_init()
