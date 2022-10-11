import argparse
import util
from util import *

from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=64,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.005,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.1,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=250,help='')
parser.add_argument('--group',type=float,default=4,help='g^t number')
parser.add_argument('--print_every',type=int,default=1000,help='')
parser.add_argument('--save',type=str,default='logs/',help='save path')
parser.add_argument('--expid',type=int,default=4,help='experiment id')
parser.add_argument('--checkpoint',type=str,default=None)

args = parser.parse_args()

def main():
    device = torch.device(args.device)

    adj_mx = get_adj_matrix('data/PEMS08/PEMS08.csv', 170)
    dataloader = generate_data('data/PEMS08/PEMS08.npz', args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    engine = trainer(scaler, args.in_dim, args.seq_length, args.nhid , args.dropout, args.learning_rate, args.weight_decay, args.device, supports, args.group)
    model = engine.model
    model.to(device)
    engine.model.load_state_dict(torch.load(args.checkpoint))
    engine.model.eval()
    outputs = []
    realy = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1,3)[:,:1,:,:]
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds)
        realy.append(testy)

    yhat = torch.cat(outputs,dim=0)
    yhat = scaler.inverse_transform(yhat)
    realy = torch.cat(realy,dim=0)

    amae = []
    amape = []
    armse = []
    print(yhat.shape, realy.shape)
    for i in range(12):
        pred = yhat[...,i]
        real = realy[...,i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(*util.metric(yhat, realy)))

if __name__ == "__main__":
    main()
