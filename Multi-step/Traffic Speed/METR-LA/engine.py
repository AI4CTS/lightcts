import torch.optim as optim
from lightcts_model import *
import util
import torch
class trainer():
    def __init__(self, scaler, in_dim, seq_length, nhid, dropout, lrate, wdecay, device, supports, group, lr_decay_rate=.98):
        self.model = ttnet(dropout=dropout, supports=supports, in_dim=in_dim, out_dim=seq_length, hid_dim=nhid,
                           group=group)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 3
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)

    def train(self, input, real_val, epoch=0):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, epoch)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        mae = util.masked_mae(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        mape = util.masked_mape(predict,real,0.0).item()
        mae = util.masked_mae(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return mae,mape,rmse
