import torch
from torchdiffeq import odeint_adjoint as odeint
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import scipy.stats as st
from tqdm import tqdm

class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.
    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]

def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()

class CNF_(nn.Module):
    def __init__(self, in_out_dim, hidden_dim) -> None:
        super().__init__()
        self.in_dim = in_out_dim + 1
        self.out_dim = in_out_dim
        self.layers = nn.ModuleList([])
        hidden_dim.insert(0, self.in_dim)
        for i in range(len(hidden_dim)-1):
            self.layers.append(
                nn.Linear(in_features=hidden_dim[i], out_features=hidden_dim[i+1])
            )
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(in_features=hidden_dim[-1], out_features=self.out_dim))

    def forward(self, t, states, require_div=True):
        z = states[0]
        bsz = z.shape[0]
        data_shape = z.shape
        t = t.reshape(-1)
        with torch.set_grad_enabled(True):

            t = self.check_t(t, bsz)
            z = z.reshape(bsz, -1)
            z.requires_grad_(True)
            v = torch.cat((z.reshape(bsz, -1), t.unsqueeze(1)), dim=1)
            for l in self.layers:
                v = l(v)

            if require_div:
                divv = -trace_df_dz(f=v, z=z)
                return (v.reshape(data_shape), divv.reshape(bsz, 1))
            else:
                return (v.reshape(data_shape),)

    def check_t(self, t, bsz):
        if len(t) == 1:
            t = t.view(-1).repeat(bsz)
            return t
        elif len(t) !=  bsz:
            print(len(t), z.shape)
            print("time t should either be of same size as the bsz, or as one single value")
            raise ValueError
        else:
            return t


def OptimalTransportVFS(z, x0, t, sigma):
    '''
        z: is the target samples
        x: is x(t)
        t: time_steps: 1D tensor
    '''
    t = t.reshape(-1)
    check_shape_t(t, x0)
    t = t.reshape([-1]+[1 for _ in range(len(z.shape[1:]))]).to(device)
    xt = t * (z - x0) + x0
    vt = z - x0
    # print('test xt: ', xt.shape, vt.shape)
    return  xt, vt

def OptimalTransportFM(z, x0, t, sigma):
    '''
        z: is the target samples
        x: is x(t)
        t: time_steps: 1D tensor
    '''
    t = t.reshape(-1)
    t = t.reshape([-1]+[1 for _ in range(len(z.shape[1:]))]).to(device)
    check_shape_t(t, x0)
    # z = z.unsqueeze(0).repeat(t.shape[0], 1, 1)
    # x0 = x0.unsqueeze(0).repeat(t.shape[0], 1, 1)
    xt = (1-(1-sigma)*t)*x0 + t*z
    vt = z-(1-sigma)*x0 / (1-(1-sigma)*t)
    # print('test xt: ', t.shape, x0.shape, z.shape, xt.shape, vt.shape)
    return  xt, vt

def check_shape_t(t, x0):
    if t.shape[0] == 1:
        return t.repeat(x0.shape[0])
    elif t.shape[0] != x0.shape[0]:
        raise ValueError
    else:
        return t

class vector_field_calculator():

    def __init__(self, time_point):
        self.pai = 3.1415926
        self.time_point = time_point
    
    def get_data_prob(self, x):
        pass
    
    def get_mu_t(self, x1):
        pass
        
    def get_sigma_t(self, x1):
        pass
        
    def get_condition_normal_distribution(self, x, mu, sigma):
        # 二元高斯分布
        # p = torch.exp((-(x-mu).transpose(0,1)*(x-mu))/(2*sigma*sigma))
        mu = mu.clone().detach().cpu().numpy()
        x = x.clone().detach().cpu().numpy()
        p = st.multivariate_normal.pdf(x, mean=mu, cov=sigma) 
        return p
         
    def get_x1_sample(self, x1_list): 
        return x1_list
        
    def sum_x1_condition(self, x, x1_list):
        sum_x1_condition_p = 0.0
        for x1 in x1_list:
            mu_t = self.get_mu_t(x1)
            sigma_t = self.get_sigma_t(x1)
            sum_x1_condition_p += self.get_data_prob(x1)*self.get_condition_normal_distribution(x, mu_t, sigma_t)
        return sum_x1_condition_p
    
    def get_condition_vertor_field(self, x, x1):
        pass
    
    def get_vector_field(self, x_point_set, x1_list):
        vector_field_value_list = {}
        vector_field_x_list = {}
        field_value = 0
        for x_ind, x_ in tqdm(enumerate(x_point_set)):
            total_weight = self.sum_x1_condition(x_, x1_list)
            for x1_ in x1_list:
                mu_t = self.get_mu_t(x1_)
                sigma_t = self.get_sigma_t(x1_)
                wieght_u = self.get_condition_normal_distribution(x_, mu_t, sigma_t)*self.get_data_prob(x1_)/total_weight
                condition_field = self.get_condition_vertor_field(x_, x1_)
                field_value += wieght_u*condition_field
            vector_field_value_list[x_ind] = field_value
            vector_field_x_list[x_ind] = (condition_field, x_, self.time_point)
        return vector_field_x_list, vector_field_value_list


class op_vfs_vector_field_calculator(vector_field_calculator):
    def __init__(self, time_point, sigma_min):
        self.pai = 3.1415926
        self.time_point = time_point
        self.sigma_min = sigma_min
        
    def get_data_prob(self, x1):
        #这里假设是取值区域上的均匀分布
        k = 0.1
        return k 
    
    def get_mu_t(self, x1):
        return self.time_point * x1
    
    def get_sigma_t(self, x1):
        return 1 - (1-self.sigma_min) * self.time_point
    
    def get_condition_vertor_field(self, x, x1):
        return (x1 - (1-self.sigma_min) * x)/(1- (1-self.sigma_min) * self.time_point)
    
    def get_x1_sample(self, x1_list): 
        return x1_list

    def get_vector_field(self, x_point_set, x1_list):
        # vector_field_value_list = {}
        # vector_field_x_list = {}
        vector_field_data_list = {}
        field_value = 0
        for x_ind, x_ in tqdm(enumerate(x_point_set)):
            total_weight = self.sum_x1_condition(x_, x1_list)
            field_value = 0
            for x1_ in x1_list:
                mu_t = self.get_mu_t(x1_)
                sigma_t = self.get_sigma_t(x1_)
                wieght_u = self.get_condition_normal_distribution(x_, mu_t, sigma_t)*self.get_data_prob(x1_)/total_weight
                condition_field_x = self.get_condition_vertor_field(x_, x1_)
                field_value += wieght_u*condition_field_x

            # vector_field_value_list[x_ind] = field_value
            vector_field_data_list[x_ind] = (x_, condition_field_x, self.time_point, field_value)
        return vector_field_data_list

    #   p1服从一定区域上的均匀分布时，并且积分值比较容易算出来的时候，此方法在一些特殊的计算下可用
    def get_vector_field_fast(self, x_point_set, x1_list):
        vector_field_value_list = {}
        vector_field_x_list = {}
        field_value = 0
        for x_ind, x_ in tqdm(enumerate(x_point_set)):
            # total_weight = self.sum_x1_condition(x_, x1_list)
            total_weight = 1.0
            for x1_ in x1_list:
                mu_t = self.get_mu_t(x1_)
                sigma_t = self.get_sigma_t(x1_)
                wieght_u = self.get_condition_normal_distribution(x_, mu_t, sigma_t)*self.get_data_prob(x1_)/total_weight
                condition_field_x = self.get_condition_vertor_field(x_, x1_)
                field_value += wieght_u*condition_field_x
            vector_field_value_list[x_ind] = field_value
            vector_field_x_list[x_ind] = (condition_field_x, x_, self.time_point)
        return vector_field_x_list, vector_field_value_list


class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]
        batchsize = z.shape[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            W, B, U = self.hyper_net(t)
            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)
            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)
            # dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)
            dlogp_z_dt = torch.tensor(0.).to(device)

        return (dz_dt, dlogp_z_dt)
