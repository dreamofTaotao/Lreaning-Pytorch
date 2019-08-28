# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:56:36 2019

@author: test
    1.现在有了前面的知识，现在可以定义一个模型，这个模型继承自nn.Module的大类。如果需要
定义一个比Sequential模型更加复杂的模型，就需要定义nn.Module模型。
   *2.了解Sequential网络模式，请参考Learning_torch_nn.py文件
"""
#课前知识：
#   1.了解一下torch.nn.Module的相关知识
#如果想要定义一个相对来说复杂一些的网络的话，可以直接继承这个类，那么，我们就可以直接进行
#定义相关的网络了。
#   2.super的相关用法：
#       【1】传统解释：传统父类方法调用
#       【2】
#       【3】解释参考：https://blog.csdn.net/wind_602/article/details/78608201
#super的总结：1.super()方法在py3.x中可用,py2.x是使用super关键字并传入父类名称和子类self对象
#            2.super()调用方法将以MRO的搜索方式进行关系链的调用
#            3.在子类中如果没有使用super()将停止关系链的调用
#            4.在MRO最右边的顶层基类中不要声明super()语句调用
#            5.混合方法使用super()仍然以上述规则来查询
#       【4】super中传入参数的详解：https://www.cnblogs.com/yanlin-10/p/10272338.html


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

#定义模型：继承nn.Module
class TwoLayerNet(torch.nn.Module):
    
    #初始化函数：
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        #define the model architecture 
        self.linear1 = torch.nn.Linear(D_in, H, bias=False)
        self.linear2 = torch.nn.Linear(H, D_out, bias=False)
        
    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred
        

#Neural NetWork Module:
module = TwoLayerNet(D_in, H, D_out)

#loss_function
loss_fn = nn.MSELoss()
#↑如果在nn.MSELoss()中加入参数：reduction='sum'可能下降有些慢【玄学】

#优化方法optim中的Adam：
optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)


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

#一个网络搭建的步骤：
#    1.定义输入输出数据
#    2.定义神经网络的模型
#    3.定义loss function，损失函数的计算函数
#    4.然后把这个模型交给optimizer去做optimize（也就是做模型的优化）
#这里的模型优化，指的就是将新的weights重新刷新。
#    5.然后进入到真正的训练网路，进行数据的训练
#        【1】Forward pass：前向传播
#        【2】计算损失值
#        【3】优化数据的清零（这里是将我们的重新清零赋值）
#        【4】loss backward后向传播（俗称求导）
#        【5】模型的更新，也就是将我们求导后的数据进行重新赋值给weights,
#        parameters中的[tensor, grad]进入到optimizer中进行计算后赋值。

