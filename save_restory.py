import torch
from torch.autograd import Variable
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt

def save():
    net=torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,2)
        )    
    # define the network
    print(net)  # net architecture

    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

    plt.ion()   # something about plotting

    for t in range(10):
        out = net(x)                 # input x and predict based on x
        loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    torch.save(net,'net.pkl')#entire net
    torch.save(net.state_dict(),'net_para.pkl')#para

def restore_net():
    net2=torch.load('net.pkl')

def restore_para():
    net3=torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,2)
        )    
    net3.load_state_dict(torch.load('net_para.pkl'))