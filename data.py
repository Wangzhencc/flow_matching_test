from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from sklearn.datasets import make_circles, make_checkerboard
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_batch_checkboard(num_samples):

    x = torch.rand(num_samples, 2)
    factor = torch.tensor([[0,0],[1,1],[-1,-1],[-2,-2],[1,-1],[0,-2],[-1,1],[-2,0]])
    x = x + factor[torch.randint(low=0, high=8, size=(num_samples,))]
    x = x.type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)
    
    return(x, logp_diff_t1)

def get_batch_circle(num_samples):

    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return(x, logp_diff_t1)

def get_batch_gaussian(num_samples, D):

    D = 10.
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
