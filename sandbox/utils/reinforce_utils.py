
import numpy as np

import torch
from torch.autograd import Variable


pi = Variable(torch.FloatTensor([math.pi])).to(device)

def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi).sqrt()
    return a*b