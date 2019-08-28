# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:42:38 2019

@author: test
本次课程中我们使用了torch:nn库来进行构建网络，用Pytorch autograd来构建计算图和计算gradient，
然后Pytorch会帮助我们自动计算gradient。
"""
#课前知识：
#   1.torch.nn的库是各种定义Neural NetWord的方法。
#   2.根据官方的解释，torch.nn.Module是一个网络的基类。
#   3.比如说我们构建的前向传播的相关的代码，y_pred = x.mm(w1).clamp(min = 0).mm(w2)
#可以进行在一个torch.nn.Sequential()中。一个时序容器。官方对于该库的解释：
#Modules 会以他们传入的顺序被添加到容器中。当然，也可以传入一个OrderedDict。
#举个例子：
#    # Example of using Sequential
#    model = nn.Sequential(
#              nn.Conv2d(1,20,5),
#              nn.ReLU(),
#              nn.Conv2d(20,64,5),
#              nn.ReLU()
#            )
#    # Example of using Sequential with OrderedDict
#    model = nn.Sequential(OrderedDict([
#              ('conv1', nn.Conv2d(1,20,5)),
#              ('relu1', nn.ReLU()),
#              ('conv2', nn.Conv2d(20,64,5)),
#              ('relu2', nn.ReLU())
#            ]))
#   4.Linear的理解：Linear是对输入数据做线性变换：y=Ax+b
#       class torch.nn.Linear(in_features, out_features, bias=True)
#   举个例子：
#            import torch.nn as nn
#            import torch
#            m = nn.Linear(20, 30)
#            input = torch.autograd.Variable(torch.randn(128, 20))
#            output = m(input)
#            print(output.size())
#            输出：torch.Size([128, 30])
#   5.torch.nn.ReLU()激活函数【activision】:对输入运用修正线性单元函数${ReLU}(x)= max(0, x)$。
#   6.class torch.nn.MSELoss(size_average=True)[source]
#        创建一个衡量输入x(模型预测输出)和目标y之间均方误差标准。
#           公式：loss(x,y)=1/n∑(xi−yi)2
#        x 和 y 可以是任意形状，每个包含n个元素。
#        对n个元素对应的差值的绝对值求和，得出来的结果除以n。
#        如果在创建MSELoss实例的时候在构造函数中传入size_average=False，那么求出来的平方和将不会除以n
#   7.zero_grad():将module中的所有模型参数的梯度设置为0.
import torch
import torch.nn as nn

N,D_in,H,D_out=64,1000,100,10

#随机创建了torch数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

#学习率
learning_rate = 1e-1

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

#epoch：500
for it in range(500):

    #Forward pass
    y_pred = module(x) #如果直接定义了module的话每一个epoch进来之后，那么将会自动在module中进行
                       #层数的计算，因为我们已经定义过了相关的层中的计算式。 
    
    #compute loss
    loss = loss_fn(y_pred, y)
    print(it, loss.item())
    
    #Backwward pass 反向传播
    #compute the gradient 梯度求导(向后看)
    loss.backward() #因为现在的模型被封装好了，但是参数是可以拿到的，但是需要一个一个的拿
    
    #update weights of w1 and w2
    with torch.no_grad():
        for param in module.parameters():#每一个param包含两个部分：(tensor, grad)
                                         #每一次我们需要将grad进行减掉
            param -= learning_rate * param
            
    #grad清零：(如果不进行清零的，那么每次的gradient会进行叠加)
    module.zero_grad()    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        