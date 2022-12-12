import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import pandas as pd
import scipy.stats as st
import matplotlib.animation as animation
import cv2
from pathlib import Path
from tqdm import tqdm
import os

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

def save_sif_sample_data(z_t_samples, x0, ts):
    for ix, data in tqdm(enumerate(z_t_samples.detach().cpu())):
        fig = plot(x0, data, M=2.)
        plt.savefig("./figs/p_{0}.png".format(ix), dpi=60)
        plt.close()

    fig = plt.figure()
    ims = []
    for i in range(ts.shape[0]):
        img = cv2.imread("./figs/p_{0}.png".format(i))
        (r, g, b) = cv2.split(img)
        img = cv2.merge([b,g,r])
        im = plt.imshow(img, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, artists=ims, interval=50)
    ani.save("animation.gif")




