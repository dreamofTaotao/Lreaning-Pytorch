# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:42:08 2019

@author: test
    本次课程中我们使用了torch:nn库来进行构建网络，用Pytorch autograd来构建计算图和计算gradient,
并且使用optim函数更新网络，然后Pytorch会帮助我们自动计算gradient。
    这一次我们使用optim函数的话，就不用了手动更新模型的weights了，而是使用
optim这个包来帮助我们的网络进行更新参数。optimal这个package提供了各种不同的模型优化方法，
包括SGD+momentum,RMSProp,Adam等等。
"""
#课前知识：
#   1.optim:中文介绍及用法详解：
#       https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/#optimizer

import torch
import torch.nn as nn

N,D_in,H,D_out=64,1000,100,10

#随机创建了torch数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

#学习率
learning_rate = 1e-4

#Neural NetWork Module:
module = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),# w1*x + b,如果不需要bias可以添加，bias = Fasle
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

#初始化weight时进行选择性初始化
torch.nn.init.normal_(module[0].weight)
torch.nn.init.normal_(module[2].weight)

#如果想要在cuda中做操作的话，需要：
#mudule = module.cuda()

#loss_function
loss_fn = nn.MSELoss()
#如果在nn.MSELoss()中加入参数：reduction='sum'可能下降有些慢【玄学】

#优化方法optim中的Adam：
#optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)

#优化方法optim中的SGD
optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)

#epoch：500
for it in range(500):

    #Forward pass
    y_pred = module(x) #如果直接定义了module的话每一个epoch进来之后，那么将会自动在module中进行
                       #层数的计算，因为我们已经定义过了相关的层中的计算式。 
    
    #compute loss
    loss = loss_fn(y_pred, y)
    print(it, loss.item())
    
    #清除网络中的权重，重新赋值
    optimizer.zero_grad()
    
    #Backwward pass 反向传播
    #compute the gradient 梯度求导(向后看)
    loss.backward() #因为现在的模型被封装好了，但是参数是可以拿到的，但是需要一个一个的拿
    
    #update model parameters
    optimizer.step()
    
