import torch
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from data import get_batch_circle, get_gaussian_pdf, vertor_field_dataset, conditional_vertor_field_dataset
from utils import RunningAverageMeter, setup_seed, plot, index_sampler
from model import CNF_, OptimalTransportVFS, OptimalTransportFM, op_vfs_vector_field_calculator
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from  sampler import ode_sampler
from torch.utils.data import DataLoader

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
###尺度信息有影响，需要 normalization 数据
sigma = 0.05 # variance of x(1) distribution
std = 0.2
viz_timesteps = 240
viz_samples = 1000


def get_raw_data():
    # circles
    samples_1, _  = get_batch_circle(num_samples)
    # 正态分布
    samples_0 = torch.randn_like(samples_1).to(device) * std

    x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]
    x_0 = samples_0.detach().clone()[torch.randperm(len(samples_1))]
    p_x0 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([0.0, 0.0]).to(device),
            covariance_matrix=torch.tensor([[std**2, 0.0], [0.0, std**2]]).to(device)
        )
    log_prob_x0 = p_x0.log_prob(x_0)

    # plot_tem(x_1, x_0)

    return x_1, x_0


def plot_tem(x_1, x_0):
    view_Samples = x_1.detach().cpu()
    plt.scatter(view_Samples[:,0], view_Samples[:,1], s=2)
    plt.show()

    view_Samples = x_0.detach().cpu()
    plt.scatter(view_Samples[:,0], view_Samples[:,1], s=2)
    plt.show()
    # print('test x1, x0: ', x_1.shape, x_0.shape)


def get_batch_interpolation_data(x_1, x_0):
    tp = torch.rand(bsz).to(device)
    idx = torch.randint(low=0, high=num_samples, size=(bsz,))
    z = x_1[idx]
    # idx = torch.randint(low=0, high=num_samples, size=(bsz,))
    # x0 = x_0[torch.randint(low=0, high=num_samples, size=(bsz,))]
    x0 = x_0[idx]
    xt, vt = OptimalTransportFM(z, x0, tp, sigma)
    # xt, vt = OptimalTransportVFS(z, x0, tp, sigma)
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
        ### 这里，注意下xt，vt
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
    return func

def train_demo_main():
    setup_seed(42)
    target_sample_num = 10000
    raw_sample_num = 10000
    batch_size = 1280

    conditional_vector_field_datasets = conditional_vertor_field_dataset(target_sample_num, raw_sample_num, sigma)
    dataloader = DataLoader(conditional_vector_field_datasets, batch_size=batch_size, shuffle=True)
    func = CNF_(in_out_dim=2, hidden_dim=hidden_dim).to(device)
    func.train()
    optimizer = Adam(func.parameters(), lr=1e-3)
    lrsc = ExponentialLR(optimizer=optimizer, gamma=0.9998)
    loss_meter = RunningAverageMeter()
    loss_list = []
    for itr in range(niter):
        for step1, (time_p, batch_xt, batch_vt) in enumerate(dataloader):
            optimizer.zero_grad()
            ### 这里，注意下xt，vt
            # xt, vt, tp = get_batch_interpolation_data(x_1, x_0)
            (vt_pre,) = func(t=time_p, states=(batch_xt,), require_div=False)
            mse_loss = nn.MSELoss()
            loss = mse_loss(vt_pre, batch_vt)
            # loss = (vt - vt_pre).view(vt.shape[0], -1).abs().pow(2).sum(dim=1)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            lrsc.step()

            loss_meter.update(loss.item())
            loss_list.append(loss.item())

            if itr % 100 == 0:
                print('Iter: {} step: {}, running avg loss: {:.4f}'.format(itr, step1, loss_meter.avg))

    plt.plot(loss_list)
    plt.show()
    return func

def train_field_data():
    setup_seed(42)
    time_delta_num = 2
    target_sample_num = 100
    raw_sample_num = 10
    batch_size = 12
    niter = 100

    vector_field_datasets = vertor_field_dataset(time_delta_num, target_sample_num, raw_sample_num)
    dataloader = DataLoader(vector_field_datasets, batch_size=batch_size, shuffle=True)
    func = CNF_(in_out_dim=2, hidden_dim=hidden_dim).to(device)
    func.train()
    optimizer = Adam(func.parameters(), lr=1e-3)
    lrsc = ExponentialLR(optimizer=optimizer, gamma=0.9998)
    loss_meter = RunningAverageMeter()
    loss_list = []
    for epoch in range(niter):
        for step1, (time_p, batch_xt, batch_vt) in enumerate(dataloader):
            # training
            optimizer.zero_grad()
            time_p = time_p.to(device).to(torch.float32)
            (vt_pre,) = func(t=time_p, states=(batch_xt,), require_div=False)
            mse_loss = nn.MSELoss()
            loss = mse_loss(vt_pre, batch_vt)
            # loss = (vt - vt_pre).view(vt.shape[0], -1).abs().pow(2).sum(dim=1)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            lrsc.step()

            loss_meter.update(loss.item())
            loss_list.append(loss.item())

            if step1 % 100 == 0:
                print('Iter: epoch: {}, step: {}, running avg loss: {:.4f}'.format(epoch, step1, loss_meter.avg))

    return func

def draw_plot(func):
    t0 = 0.
    t1 = 1.
    # sigma = 0.001 # variance of x(1) distribution
    var = 0.05
    viz_timesteps = 5
    x0 = torch.randn(3000, 2)* sigma

    ts = torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device)

    z_t_samples, _  = ode_sampler(func, x0, ts)

    z, logp_diff_t1 = get_batch_circle(3000)
    z = z.cpu()
    plt.scatter(z[:,0], z[:,1])
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.show()
    for ix, data in enumerate(z_t_samples.detach().cpu()):
        if ix % 100 == 0:
            plt.scatter(data[:,0], data[:,1])
            plt.title("t="+str(int(ts[ix])))
            plt.xlim([-1.5, 1.5])
            plt.ylim([-1.5, 1.5])
            plt.show()

if __name__ == "__main__":
    # func = train_field_data()
    func = train_demo_main()
    draw_plot(func)

