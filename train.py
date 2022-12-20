import os
import torch
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from data import get_batch_circle, get_batch_gaussian, get_trip_data, get_batch_checkboard, get_gaussian_pdf, vertor_field_dataset, conditional_vertor_field_dataset, ops_vertor_field_dataset
from utils import RunningAverageMeter, setup_seed, sample_plot, index_sampler, save_sif_sample_data, draw_plot
from model import MLP, RectifiedFlow, CNF_, OptimalTransportVFS, OptimalTransportFM, op_vfs_vector_field_calculator, op_ops_vector_field_calculator
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from  sampler import ode_sampler
from torch.utils.data import DataLoader
import copy 

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

def get_trip_raw_data(outputname = None):

    samples_0, samples_1  = get_batch_gaussian(num_samples, 5)
    x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]
    x_0 = samples_0.detach().clone()[torch.randperm(len(samples_1))]
    if outputname:
        sample_plot(samples_0, samples_1, 15.0, outputname)

    return x_1.to(device), x_0.to(device)


def get_checkboard_raw_data(outputname = None):

    samples_1, _  = get_batch_checkboard(num_samples)
    # 正态分布
    samples_0 = torch.randn_like(samples_1).to(device) * std

    x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]
    x_0 = samples_0.detach().clone()[torch.randperm(len(samples_1))]
    p_x0 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([0.0, 0.0]).to(device),
            covariance_matrix=torch.tensor([[std**2, 0.0], [0.0, std**2]]).to(device)
        )
    log_prob_x0 = p_x0.log_prob(x_0)
    if outputname:
        sample_plot(samples_0, samples_1, 2.0, outputname)

    return x_1.to(device), x_0.to(device)

def get_raw_data(outputname = None):
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
    if outputname:
        sample_plot(samples_0, samples_1, 2.0, outputname)

    return x_1.to(device), x_0.to(device)


def plot_tem(x_1, x_0):
    view_Samples = x_1.detach().cpu()
    plt.scatter(view_Samples[:,0], view_Samples[:,1], s=2)
    plt.show()

    view_Samples = x_0.detach().cpu()
    plt.scatter(view_Samples[:,0], view_Samples[:,1], s=2)
    plt.show()
    # print('test x1, x0: ', x_1.shape, x_0.shape)


def get_batch_interpolation_data(x_1, x_0):

    ###奇怪的不同
    tp = torch.rand(bsz).to(device)
    idx = torch.randint(low=0, high=num_samples, size=(bsz,))
    z = x_1[idx]
    # idx = torch.randint(low=0, high=num_samples, size=(bsz,))
    # x0 = x_0[torch.randint(low=0, high=num_samples, size=(bsz,))]
    x0 = x_0[idx]
    xt, vt = OptimalTransportFM(z, x0, tp, sigma)
    # xt, vt = OptimalTransportVFS(z, x0, tp, sigma)
    return xt, vt, tp


def train_trip_point_main(outputname):
    setup_seed(42)
    x_1, x_0 = get_trip_raw_data(outputname) 
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


def train_dataloader_main():
    setup_seed(42)
    time_delta_num = 100
    target_sample_num = 100
    raw_sample_num = 100
    batch_size = 1280
    niter = 1000

    vector_field_datasets = ops_vertor_field_dataset(time_delta_num, target_sample_num, raw_sample_num)
    dataloader = DataLoader(vector_field_datasets, batch_size=batch_size, shuffle=True)

    rectified_flow_1 = RectifiedFlow(model=MLP(2, hidden_num=128), num_steps=100)
    optimizer = torch.optim.Adam(rectified_flow_1.model.parameters(), lr=5e-3)

    loss_curve = []
    loss_meter = RunningAverageMeter()
    for i in tqdm(range(niter+1)):
        for step1, (time_p, batch_xt, batch_vt) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = rectified_flow.model(batch_xt, time_p)
            loss = (batch_vt - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
            loss = loss.mean()
            loss.backward()

            optimizer.step()
            loss_meter.update(loss.item())
            loss_curve.append(np.log(loss.item())) ## to store the loss curve
            if i % 100 == 0:
                print('Iter: {}, running avg loss: {:.4f}'.format(i, loss_meter.avg))
    
    return rectified_flow, loss_curve

def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, niter):
    loss_curve = []
    loss_meter = RunningAverageMeter()
    for i in tqdm(range(niter+1)):
        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone().to(device)
        z1 = batch[:, 1].detach().clone().to(device)
        z_t, t, target = rectified_flow.get_train_tuple(z0=z0, z1=z1)
        pred = rectified_flow.model(z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()

        optimizer.step()
        loss_meter.update(loss.item())
        loss_curve.append(np.log(loss.item())) ## to store the loss curve
        if i % 100 == 0:
            print('Iter: {}, running avg loss: {:.4f}'.format(i, loss_meter.avg))

    return rectified_flow, loss_curve


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
    time_delta_num = 100
    target_sample_num = 100
    raw_sample_num = 100
    batch_size = 1280
    niter = 1000

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

def infer_draw_plot(func, x0, out_file):
    t0 = 0.
    t1 = 1.
    # sigma = 0.001 # variance of x(1) distribution
    var = 0.05
    viz_timesteps = 100
    # x0 = torch.randn(3000, 2)* sigma
    out_file = 'figs/'+out_file
    if not os.path.exists(out_file):
        os.makedirs(out_file)

    ts = torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device)
    z_t_samples, _  = ode_sampler(func, x0, ts)
    save_sif_sample_data(z_t_samples, x0, ts, out_file)

def train_with_rec_flow(samples_0, samples_1, samples_00, samples_11):
    x_0 = samples_0.detach().clone()[torch.randperm(len(samples_0))]
    x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]
    x_pairs = torch.stack([x_0, x_1], dim=1)

    rectified_flow_1 = RectifiedFlow(model=MLP(2, hidden_num=128), num_steps=100)
    optimizer = torch.optim.Adam(rectified_flow_1.model.parameters(), lr=5e-3)

    rectified_flow_1, loss_curve = train_rectified_flow(rectified_flow_1, optimizer, x_pairs, batch_size, niter)
      
    draw_plot(rectified_flow_1, z0=samples_00, z1=samples_11, outfile=out_file2, M = 2, N=100)
    return rectified_flow_1

def get_target_neighbor(z1, z11):
    z2 = z1.detach().clone()
    for idx, z_item in enumerate(z1):
        z_idx = get_target_idx(z_item, z11)
        z2[idx] = z11[z_idx]
    return z2

def get_target_idx(z_item, z11):
    delta_z = torch.sum((z_item - z11)*(z_item - z11), dim = 1)
    z_idx = torch.argmin(delta_z)
    return z_idx


def re_train_with_rec_flow(rectified_flow_1, samples_0, samples_1, samples_00, samples_11, out_file2):

    z10 = samples_0.detach().clone()
    traj = rectified_flow_1.sample_ode(z0=z10.detach().clone(), N=100)
    # print('test traj: ', len(traj[0]), traj[0].shape)
    z11 = traj[-1].detach().clone()
    z1 = samples_1.detach().clone()
    # z_pairs = torch.stack([z10, z11], dim=1)
    z2 = get_target_neighbor(z1, z11)
    z_pairs = torch.stack([z10, z2], dim=1)
    rectified_flow_2 = RectifiedFlow(model=MLP(2, hidden_num=128), num_steps=100)
    # rectified_flow_2.net = copy.deepcopy(rectified_flow_1) # we fine-tune the model from 1-Rectified Flow for faster training.
    optimizer = torch.optim.Adam(rectified_flow_2.model.parameters(), lr=5e-3)
    rectified_flow_2, loss_curve = train_rectified_flow(rectified_flow_2, optimizer, z_pairs, batch_size, niter)      
    draw_plot(rectified_flow_2, z0=samples_00, z1=samples_11, outfile=out_file2, M = 2,  N=100)
    # save_sif_sample_data()
    return rectified_flow_2
   

if __name__ == "__main__":
    # func = train_field_data()
    # func = train_demo_main()
    # func = train_main()
    batch_size = 4096
    outputname = 'checkboard_point2_2'
    out_file2 = 'figs/checkboard_point2_2'
    out_file3 = 'figs/checkboard_point3_2'
    out_file4 = 'figs/checkboard_point4_2'
    # samples_1, samples_0 = get_checkboard_raw_data(outputname)
    # samples_11, samples_00 = get_checkboard_raw_data() 
    # get_target_neighbor(samples_1, samples_11)
    # print('test z1 shape: ', samples_1.shape)
    # # func = train_demo_main()
    # func = train_trip_point_main(outputname)
    # _, x00 = get_trip_raw_data(outputname)
    # infer_draw_plot(func, x00, out_file)

    # rectified_flow_1 = train_with_rec_flow(samples_0, samples_1, samples_00, samples_11)


    # rectified_flow_1 = train_with_rec_flow(samples_0, samples_1, samples_00, samples_11)
    # rectified_flow_2 = re_train_with_rec_flow(rectified_flow_1, samples_0, samples_1, samples_00, samples_11, out_file3)
    # re_train_with_rec_flow(rectified_flow_2, samples_0, samples_1, samples_00, samples_11, out_file4)
 
    train_dataloader_main()