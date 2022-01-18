import sys
sys.path.append("../mutation_analysis_by_DBT")

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import data_generators.utils as Utils


class Net1(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.criterion = nn.MSELoss()

        self.wconv1=nn.Sequential(
            nn.Conv1d(15, 128, 3, padding=1),
            nn.ReLU()
        )
        self.wconv2=nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.mconv1=nn.Sequential(
            nn.Conv1d(15, 128, 3, padding=1),
            nn.ReLU()
        )
        self.mconv2=nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.dense1=nn.Sequential(
            nn.Linear(600, 256),
            nn.ReLU()
        )
        self.dense2=nn.Sequential(
            nn.Linear(5120, 256),
            nn.ReLU()
        )
        self.dense3=nn.Sequential(
            nn.Linear(2560, 256),
            nn.ReLU()
        )
        self.dense4=nn.Sequential(
            nn.Linear(768,64)
        )
        self.regressor=nn.Sequential(
            nn.Linear(64,1)
        )
        self.softsign=nn.Softsign()
    
    def get_features(self, wild_enc, mutant_enc):
        # both inp size: batch, 15, 20
        wild_conv1 = self.wconv1(wild_enc)#out:batch,128,20
        wild_conv2 = self.wconv2(wild_conv1)#out:batch,128,10
        mut_conv1 = self.mconv1(mutant_enc)#out:batch,128,20
        mut_conv2 = self.mconv2(mut_conv1)#out:batch,128,10
        flat_wild_0 = wild_enc.view(wild_enc.size(0), -1)#300
        flat_wild_1 = wild_conv1.view(wild_conv1.size(0), -1)#2560
        flat_wild_2 = wild_conv2.view(wild_conv2.size(0), -1)#1280
        flat_mut_0 = mutant_enc.view(mutant_enc.size(0), -1)#300
        flat_mut_1 = mut_conv1.view(mut_conv1.size(0), -1)#2560
        flat_mut_2 = mut_conv2.view(mut_conv2.size(0), -1)#1280
        cat1_1 = torch.cat((flat_wild_0, flat_mut_0), 1)#600
        cat1_2 = torch.cat((flat_wild_1, flat_mut_1), 1)#5120
        cat1_3 = torch.cat((flat_wild_2, flat_mut_2), 1)#2560
        fc1_1 = self.dense1(cat1_1)#256
        fc1_2 = self.dense2(cat1_2)#256
        fc1_3 = self.dense3(cat1_3)#256
        cat_2 = torch.cat((fc1_1, fc1_2, fc1_3), 1)#768
        features = self.dense4(cat_2)#64
        return features

    def forward(self, wild_enc, mutant_enc):
        features = self.get_features(wild_enc, mutant_enc)
        out = self.softsign(self.regressor(features))
        return out

    def run_batch(self, batch_df):
        losses = []
        trgt_ddg_list, pred_ddg_list = [], []
        for row in batch_df.itertuples(index=False):
            # loading the input features 
            wild_enc, mutant_enc = Utils.load_pickle("data/encoded/"+row.wild+".pkl"), Utils.load_pickle("data/encoded/"+row.mutant+".pkl")
            # converting into tensor
            ddg_trgt = torch.tensor([row.ddg/10], dtype=torch.float32).unsqueeze(0).to(self.device)
            wild_enc_tnsr = torch.tensor(wild_enc).unsqueeze(0).to(self.device)
            mutant_enc_tnsr = torch.tensor(mutant_enc).unsqueeze(0).to(self.device)
            # print(wild_enc_tnsr.dtype, wild_enc_tnsr.shape, mutant_enc_tnsr.dtype, mutant_enc_tnsr.shape, ddg_trgt.dtype, ddg_trgt.shape)

            ddg_pred = self.forward(wild_enc_tnsr, mutant_enc_tnsr)
            trgt_ddg_list.append(ddg_trgt[0].item()), pred_ddg_list.append(ddg_pred[0].item())
            # print(ddg_pred, ddg_trgt)
            loss = self.criterion(ddg_pred, ddg_trgt)
            # print(loss.item())
            losses.append(loss)
            # break
        batch_loss = torch.stack(losses).mean()
        return batch_loss, trgt_ddg_list, pred_ddg_list

    def train_loop(self, train_loader, optimizer):
        self.train()
        losses = []
        for batch_no, batch_df in enumerate(train_loader):
            batch_loss, trgt_ddg_list, pred_ddg_list = self.run_batch(batch_df)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            losses.append(batch_loss.item())
            # print("batch_no:{}, batch_shape:{}, batch_loss:{}".format(batch_no, batch_df.shape, batch_loss))
            # break
        epoch_loss = np.mean(losses)
        return epoch_loss

            

    def validate(self, val_loader):
        self.eval()
        loss, trgt_ddg_list, pred_ddg_list = self.run_batch(val_loader)
        # print(loss)
        return loss.item(), trgt_ddg_list, pred_ddg_list


# sample usage
# import pandas as pd
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = Net1(device)
# model.to(device)
# batch_df = pd.read_csv("data/datasets/train.csv")
# model.run_batch(batch_df)
