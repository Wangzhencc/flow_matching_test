import numpy as np
import matplotlib.pyplot as plt
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ode_sampler(func, x0, ts):
    ts = ts.type_as(x0)
    z_t_samples = odeint(
                    func,
                    x0.to(device),
                    ts,
                    # (x0.to(device), torch.zeros(x0.shape[:1]).to(device)),
                    atol=1e-7,
                    rtol=1e-7,
                    method='dopri5',
                )
    return z_t_samples
