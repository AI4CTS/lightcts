KD_PEMS08.pyimport math

import torch.optim as optim
from model_kd import *
import util
from STG_model import *


class TORKD_func(nn.Module):
    """Distilling the Knowledge for Regression"""
    def __init__(self, MAD=6, batch_size=32, alpha=2):
        super(TORKD_func, self).__init__()
        self.MAD = MAD
        self.beta = batch_size
        self.alpha = alpha


    def forward(self, y_s, y_t, y_true):
        print('y_s',y_s.size())
        print('y_t',y_t.size())
        print('y_true',y_true.size())
        abstrt=abs(y_true-y_t)
        sigma=1.4826*self.MAD
        eoutlier=sigma*sqrt(-2*math.log10(2.506628*sigma*self.alpha/self.beta))
        if abstrt < eoutlier:
            loss=(y_s-y_true)*(y_s-y_true)
        else:
            loss=sqrt(abs(y_s-y_t))

        return loss




class FSP(nn.Module):
    def __init__(self):
        super(FSP, self).__init__()

    def forward(self, g_s, g_t):
        s_fsp = self.compute_fsp(g_s)
        t_fsp = self.compute_fsp(g_t)
        loss_group = [self.compute_loss(s, t) for s, t in zip(s_fsp, t_fsp)]
        return loss_group[0]

    @staticmethod
    def compute_loss(s, t):
        return (s - t).pow(2).mean()

    @staticmethod
    def compute_fsp(g):
        fsp_list = []
        for i in range(len(g) - 1):
            bot, top = g[i], g[i + 1]
            b_H, t_H = bot.shape[2], top.shape[2]
            if b_H > t_H:
                bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
            elif b_H < t_H:
                top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                pass
            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)
            bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
            top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)

            fsp = (bot * top).mean(-1)
            fsp_list.append(fsp)
        return fsp_list
fsp=FSP()

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.t_model = STGModel(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.t_model.to(device)

        self.s_model = STGModel(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.s_model.to(device)

        self.t_model.eval()

        self.alpha = 5
        self.beta = 5

        lr_decay_rate=.98

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.s_model.parameters()), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 3
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)

    def train(self, input, real_val):
        self.s_model.train()
        self.optimizer.zero_grad()

        DistillTOR = TORKD_func()

        with torch.no_grad():
            t_preds  = self.t_model(input)

        output = self.s_model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val,dim=1)


        DistillTOR(t_preds,output,real)




        predict = self.scaler.inverse_transform(output)


        loss.backward()


        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.s_model.parameters(), self.clip)
        self.optimizer.step()

        mape = util.masked_mape(predict, real, 0.0).item()
        mae = util.masked_mae(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return mae, mape, rmse, output

    def eval(self, input):
        self.s_model.eval()
        output, = self.s_model(input)


        return output
