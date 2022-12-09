import torch
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from data import get_batch_circle
from utils import RunningAverageMeter, setup_seed, plot
from model import CNF_, OptimalTransportVFS, OptimalTransportFM
import matplotlib.pyplot as plt
import torch.nn as nn


# parameters
t0 = 0.
t1 = 1.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
niter = 10000
num_samples = 10000
bsz = 1280
n_x = 1

hidden_dim = [100, 256, 256, 100]
width = 64
sigma = 0.05 # variance of x(1) distribution
std = 0.2
viz_timesteps = 240
viz_samples = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_raw_data():
    # circles
    samples_1, _  = get_batch_circle(num_samples)
    samples_0 = torch.randn_like(samples_1).to(device) * std

    x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]

    x_0 = samples_0.detach().clone()[torch.randperm(len(samples_1))]
    p_x0 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([0.0, 0.0]).to(device),
            covariance_matrix=torch.tensor([[std**2, 0.0], [0.0, std**2]]).to(device)
        )
    log_prob_x0 = p_x0.log_prob(x_0)

    # view_Samples = x_1.detach().cpu()
    # plt.scatter(view_Samples[:,0], view_Samples[:,1], s=2)
    # plt.show()

    # view_Samples = x_0.detach().cpu()
    # plt.scatter(view_Samples[:,0], view_Samples[:,1], s=2)
    # plt.show()
    # print('test x1, x0: ', x_1.shape, x_0.shape)
    return x_1, x_0

def get_batch_interpolation_data(x_1, x_0):
    tp = torch.rand(bsz).to(device)
    idx = torch.randint(low=0, high=num_samples, size=(bsz,))
    z = x_1[idx]
    # idx = torch.randint(low=0, high=num_samples, size=(bsz,))
    # x0 = x_0[torch.randint(low=0, high=num_samples, size=(bsz,))]
    x0 = x_0[idx]
    xt, vt = OptimalTransportFM(z, x0, tp, sigma)
    return xt, vt, tp


def train_main():
    setup_seed(42)
    x_1, x_0 = get_raw_data()
    func = CNF_(in_out_dim=2, hidden_dim=hidden_dim).to(device)
    func.train()
    optimizer = Adam(func.parameters(), lr=1e-3)
    lrsc = ExponentialLR(optimizer=optimizer, gamma=0.9998)
    loss_meter = RunningAverageMeter()
    loss_list = []
    for itr in range(niter):
        optimizer.zero_grad()
        xt, vt, tp = get_batch_interpolation_data(x_1, x_0)
        (vt_pre,) = func(t=tp, states=(xt,), require_div=False)
        mse_loss = nn.MSELoss()
        loss = mse_loss(vt_pre, vt)
        # loss = (vt - vt_pre).view(vt.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        lrsc.step()

        loss_meter.update(loss.item())
        loss_list.append(loss.item())

        if itr % 100 == 0:
            print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg))

    plt.plot(loss_list)
    plt.show()
if __name__ == "__main__":
    train_main()

