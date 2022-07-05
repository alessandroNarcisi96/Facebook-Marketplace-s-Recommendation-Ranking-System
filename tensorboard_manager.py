from numpy import nonzero
import torch
from torch.utils.tensorboard import SummaryWriter

writer = None
def bootstrap():
    global writer 
    writer = SummaryWriter() 
    x = torch.arange(-5, 5, 0.1).view(-1, 1)
    y = -5 * x + 0.1 * torch.randn(x.size())
    

def add_scalar(title,loss,epoch):
    writer.add_scalar(title, loss, epoch)


def flush():
    writer.flush()