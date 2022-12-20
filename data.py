from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from sklearn.datasets import make_circles, make_checkerboard
from model import CNF_, OptimalTransportVFS, OptimalTransportFM, op_vfs_vector_field_calculator, op_ops_vector_field_calculator
from torch.utils import data
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from tqdm import tqdm

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

def get_batch_checkboard(num_samples):

    x = torch.rand(num_samples, 2)
    factor = torch.tensor([[0,0],[1,1],[-1,-1],[-2,-2],[1,-1],[0,-2],[-1,1],[-2,0]])
    x = x + factor[torch.randint(low=0, high=8, size=(num_samples,))]
    x = x.type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)
    
    return(x, logp_diff_t1)

def get_batch_circle(num_samples):

    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.2)
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return(x, logp_diff_t1)

def get_gaussian_pdf(num_samples, mu = (0.0,0.0), sigma=1.0):

    x_data = np.random.uniform(-2.0,2.0,(num_samples, 2))
    y_data = st.multivariate_normal.pdf(x_data, mean=mu, cov=sigma) 
    return x_data, y_data

def get_trip_data(num_samples):

    return 

def get_batch_gaussian(num_samples, D):

    VAR = 0.3
    DOT_SIZE = 4
    COMP = 3

    initial_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)]))
    initial_comp = MultivariateNormal(torch.tensor([[D * np.sqrt(3) / 2., D / 2.], [-D * np.sqrt(3) / 2., D / 2.], [0.0, - D * np.sqrt(3) / 2.]]).float(), VAR * torch.stack([torch.eye(2) for i in range(COMP)]))
    initial_model = MixtureSameFamily(initial_mix, initial_comp)
    samples_0 = initial_model.sample([num_samples])
    target_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)]))
    target_comp = MultivariateNormal(torch.tensor([[D * np.sqrt(3) / 2., - D / 2.], [-D * np.sqrt(3) / 2., - D / 2.], [0.0, D * np.sqrt(3) / 2.]]).float(), VAR * torch.stack([torch.eye(2) for i in range(COMP)]))
    target_model = MixtureSameFamily(target_mix, target_comp)
    samples_1 = target_model.sample([num_samples])

    return samples_0, samples_1

class vertor_field_dataset(data.Dataset):
    def __init__(self, time_delta_num, target_sample_num, raw_sample_num):

        self.target_sample_num = target_sample_num
        self.time_delta_num = time_delta_num
        self.raw_sample_num = raw_sample_num
        self.target_data, self.raw_data = self.get_target_sample_data()
        self.t_dict, self.vector_data_sampler = self.get_vector_field_sampler()
    def __getitem__(self,index):
        time_index = index % self.time_delta_num
        point_index = index // self.time_delta_num
        u_data = self.vector_data_sampler[time_index][point_index][1]
        u_label = self.vector_data_sampler[time_index][point_index][3]
        return self.t_dict[time_index], u_data, u_label

    def __len__(self):
        return self.time_delta_num*self.raw_sample_num
    
    def get_vector_field_sampler(self):
        self.target_sample_cal = self.target_data[:self.target_sample_num]
        t_list = [t/self.time_delta_num for t in range(self.time_delta_num)]
        ind_list = [i for i in range(self.time_delta_num)]
        t_dict = dict(zip(ind_list, t_list))

        vector_data_sampler = {}
        for i,t in tqdm(enumerate(t_list)):
            vfs_calculator = op_vfs_vector_field_calculator(t, sigma)
            vector_field_data_list = vfs_calculator.get_vector_field(self.raw_data, self.target_sample_cal)
            vector_data_sampler[i] = vector_field_data_list
        return t_dict, vector_data_sampler
    
    def get_target_sample_data(self):
        # circles
        target_data, _  = get_batch_circle(self.raw_sample_num)
        raw_data = torch.randn_like(target_data).to(device) * std

        x_1 = target_data.detach().clone()[torch.randperm(len(target_data))]
        x_0 = raw_data.detach().clone()[torch.randperm(len(raw_data))]
        
        return x_1, x_0


class ops_vertor_field_dataset(data.Dataset):
    def __init__(self, time_delta_num, target_sample_num, raw_sample_num):

        self.target_sample_num = target_sample_num
        self.time_delta_num = time_delta_num
        self.raw_sample_num = raw_sample_num
        self.target_data, self.raw_data = self.get_target_sample_data()
        self.t_dict, self.vector_data_sampler = self.get_vector_field_sampler()
    def __getitem__(self,index):
        time_index = index % self.time_delta_num
        point_index = index // self.time_delta_num
        u_data = self.vector_data_sampler[time_index][point_index][1]
        u_label = self.vector_data_sampler[time_index][point_index][3]
        return self.t_dict[time_index], u_data, u_label

    def __len__(self):
        return self.time_delta_num*self.raw_sample_num
    
    def get_vector_field_sampler(self):
        self.target_sample_cal = self.target_data[:self.target_sample_num]
        t_list = [t/self.time_delta_num for t in range(self.time_delta_num)]
        ind_list = [i for i in range(self.time_delta_num)]
        t_dict = dict(zip(ind_list, t_list))

        vector_data_sampler = {}
        for i,t in tqdm(enumerate(t_list)):
            vfs_calculator = op_ops_vector_field_calculator(t, sigma)
            vector_field_data_list = vfs_calculator.get_vector_field(self.raw_data, self.target_sample_cal)
            vector_data_sampler[i] = vector_field_data_list
        return t_dict, vector_data_sampler
    
    def get_target_sample_data(self):
        # circles
        target_data, _  = get_batch_circle(self.raw_sample_num)
        raw_data = torch.randn_like(target_data).to(device) * std

        x_1 = target_data.detach().clone()[torch.randperm(len(target_data))]
        x_0 = raw_data.detach().clone()[torch.randperm(len(raw_data))]
        
        return x_1, x_0

class conditional_vertor_field_dataset(data.Dataset):
    def __init__(self, target_sample_num, raw_sample_num, sigma):

        self.target_sample_num = target_sample_num
        self.raw_sample_num = raw_sample_num
        self.sigma = sigma
        self.target_data, self.raw_data = self.get_target_sample_data()
        self.tp, self.conditional_vector_data_sampler = self.get_conditional_vector_field_sampler()

    def __getitem__(self,index):

        u_data = self.conditional_vector_data_sampler[0][index]
        u_label = self.conditional_vector_data_sampler[1][index]
        return self.tp[index], u_data, u_label

    def __len__(self):
        return self.raw_sample_num
    
    def get_conditional_vector_field_sampler(self):
        tp = torch.rand(self.raw_sample_num).to(device)
        idx = torch.randint(low=0, high=self.raw_sample_num, size=(self.raw_sample_num,))
        z = self.target_data[idx]
        x0 = self.raw_data[idx]
        xt, vt = OptimalTransportFM(z, x0, tp, sigma)
        return tp, (xt, vt)
    
    def get_target_sample_data(self):
        # circles
        # target_data, _  = get_batch_circle(self.raw_sample_num)
        # raw_data = torch.randn_like(target_data).to(device) * std

        samples_0, samples_1  = get_batch_gaussian(num_samples, 10)
        target_data, raw_data = samples_1.to(device), samples_0.to(device)
        x_1 = target_data.detach().clone()[torch.randperm(len(target_data))]
        x_0 = raw_data.detach().clone()[torch.randperm(len(raw_data))]
        
        return x_1, x_0


