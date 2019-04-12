import torch
from torch.autograd import Variable
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt

#x=np.linspace(-5,5,100)
#unsqueeze将一维数据变为二维
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())
x,y=Variable(x),Variable(y)

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(net,self).__init__()
        self.hidden=torch.nn.Linear(n_features,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=f.relu(self.hidden(x))
        x=self.predict(x)
        return x
net_=net(1,10,1)
print(net_)

plt.ion()
plt.show()
optimizer=torch.optim.SGD(net_.parameters(),lr=0.5)

loss_func=torch.nn.MSELoss()

for t in range(100):
    #调用net类的__call__方法
    predict=net_(x)

    loss=loss_func(predict,y)

    #梯度归零
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    if t%5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),predict.data.numpy(),'r-',lw=5)
        plt.pause(0.5)
    print(loss)
plt.ioff()
plt.show()