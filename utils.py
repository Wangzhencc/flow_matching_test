import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import pandas as pd
import scipy.stats as st


class RunningAverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def plot(samples_0, samples_1, M):
    fig = plt.figure(figsize=(4,4))
    plt.xlim(-M,M)
    plt.ylim(-M,M)
    plt.title(r'Samples from $\pi_0$ and $\pi_1$')
    plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(), alpha=0.5, label=r'$\pi_0$')
    plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(), alpha=0.5, label=r'$\pi_1$')
    plt.legend()
    plt.tight_layout()
    return fig


def index_sampler(sample_size, sample_scope):
    sample_ind = torch.randint(low=sample_scope[0], high=sample_scope[1], size=sample_size) 
    return sample_ind  

def save_model():
    pass