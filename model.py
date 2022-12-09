import torch
from torchdiffeq import odeint_adjoint as odeint
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            
            if len(t) == 1:
                t = t.view(-1).repeat(bsz)
            elif len(t) !=  bsz:
                print(len(t), z.shape)
                print("time t should either be of same size as the bsz, or as one single value")
                raise ValueError
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

def OptimalTransportVFS(z, x0, t, sigma):
    '''
        z: is the target samples
        x: is x(t)
        t: time_steps: 1D tensor
    '''
    t = t.reshape(-1)
    if t.shape[0] == 1:
        t = t.repeat(x0.shape[0])
    elif t.shape[0] != x0.shape[0]:
        raise ValueError

    t = t.reshape([-1]+[1 for _ in range(len(z.shape[1:]))]).to(device)
    # x = torch.randn(list(z.shape)).unsqueeze(0).repeat(t.shape[0], 1, 1).to(device)
    # xt = (std-(std-sigma)*t)*x + (t*z+(1-t)*x0)
    # z = z.unsqueeze(0).repeat(t.shape[0], 1, 1)
    # x0 = x0.unsqueeze(0).repeat(t.shape[0], 1, 1)
    # vt = std * z - (std-sigma)*x + x0 * (2*std*(t-1)+sigma*(1-2*t))
    # vt = vt / (std-(std-sigma)*t)

    xt = t * (z - x0) + x0
    vt = z - x0

    # x = torch.randn([t.shape[0]]+list(z.shape)).to(device)
    # xt = (std-(std-sigma)*t)*x + t*z
    
    # vt = std * z.unsqueeze(0) - (std-sigma)*x
    # vt = vt / (std-(std-sigma)*t)
    return  xt, vt


class vector_field_calculator():

    def __init__(self, sample_num, time_point):
        self.pai = 3.1415926
        self.sample_num = point_num
        self.time_point = time_point
    
    def get_data_prob(self, x):
        pass
    
    def get_mu_t(self, x1):
        pass
        
    def get_sigma_t(self, x1):
        pass
        
    def get_condition_normal_distribution(self, x, mu, sigma):
        p = torch.exp(-(x-mu).transpose(0,1)*(x-mu)/(2*sigma*sigma))
        return p/torch.sqrt(2*self.pai*sigma*sigma)
         
    def get_x1_sample(self):
        pass
        
    def sum_x1_condition(self, x, x1_list):
        sum_x1_condition_p = 0.0
        for x1 in x1_list:
            mu_t = self.get_mu_t(x1)
            sigma_t = self.get_sigma_t(x1)
            sum_x1_condition_p += self.get_data_prob(x1)*self.get_condition_normal_distribution(x1, mu_t, sigma_t)
        return sum_x1_condition_p
    
    def get_condition_vertor_field(self, x, x1):
        pass
    
    def get_vector_field(self, x_point_set):
        vector_field_list = {}
        field_value = 0
        x1_list = self.get_x1_sample()
        for x_ind, x_ in enumerate(x_point_set):
            total_weight = self.sum_x1_condition(x_, x1_list)
            for x1_ in x1_list:
                mu_t = self.get_mu_t(x1_)
                sigma_t = self.get_sigma_t(x1_)
                wieght_u = self.get_condition_normal_distribution(x_, mu_t, sigma_t)*self.get_data_prob(x1)/total_weight
                condition_field = self.get_condition_vertor_field(x_, x1_)
                field_value += wieght_u*condition_field
            vector_field_list[x_ind] = field_value
        return vector_field_list


class op_vfs_vector_field_calculator(vector_field_calculator):
    def __init__(self, sample_num, time_point, sigma_min):
        self.pai = 3.1415926
        self.sample_num = point_num
        self.time_point = time_point
        self.sigma_min = sigma_min
        
    def get_data_prob(self, x1):
        return x1 #这里需要依据分布修改
    
    def get_mu_t(self, x1):
        return self.time_point * x1
    
    def get_sigma_t(self, x1):
        return 1 - (1-self.sigma_min) * self.time_point
    
    def get_condition_vertor_field(self, x, x1):
        return (x1 - (1-self.sigma_min) * x)/(1- (1-self.sigma_min) * self.time_point)
    
    def get_x1_sample(self):
        x1_list = torch.rand(self.sample_num)  
        return x1_list

def OptimalTransportFM(z, x0, x, t, sigma):
    '''
        z: is the target samples
        x: is x(t)
        t: time_steps: 1D tensor
    '''
    t = t.reshape(-1)
    if t.shape[0] == 1:
        t = t.repeat(x0.shape[0])
    elif t.shape[0] != x0.shape[0]:
        raise ValueError

    t = t.reshape([-1]+[1 for _ in range(len(z.shape[1:]))]).to(device)

    x = x.unsqueeze(0).repeat(t.shape[0], 1, 1)
    xt = (std-(std-sigma)*t)*x + (t*z+(1-t)*x0)
    z = z.unsqueeze(0).repeat(t.shape[0], 1, 1)
    x0 = x0.unsqueeze(0).repeat(t.shape[0], 1, 1)
    vt = std * z - (std-sigma)*x + x0 * (2*std*(t-1)+sigma*(1-2*t))
    vt = vt / (std-(std-sigma)*t)

    return  xt, vt