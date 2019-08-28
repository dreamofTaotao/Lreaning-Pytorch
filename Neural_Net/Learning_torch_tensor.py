# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:31:40 2019

@author: test

本次使用Pytorch tensor创建向前神经网络，计算损失以及反向求导。
在本次的代码中则是进行pytorch搭建神经网络
在Pytorch中的很多计算类型与Numpy是完全不相同的，比如说矩阵的创建，ReLU函数等等需要更改。

目标：神经网络学习
基本的pytorch学习
热身：使用pytorch实现两层神经网络
一个全连接ReLU神经网络，一个隐藏层，没有bias(偏移量),用来从x预测y,使用L2 Loss.
h=w_1*x+b_1——>第一层
a=max(0,h)——>ReLU层
y_hat=w_2*a+b_2——>第二层
在本次的神经网络是取消了bias，只是理解神经网络的基本构成
    *forward pass 前向传播
    *loss 损失值
    *backward pass 反向传播 
在Learning_torch_numpy中，是完全使用numpy来计算前向神经网络，loss和反向传播.

"""

#这里需要很多数学的知识
#   1.导数基础——链式求导法则
#   2.导数的意义——值的含义，就是每个值代表的什么
#   3.矩阵的运算基础，内积(点积)【这里需要注意的问题是矩阵的点积是不能进行前后变幻的，就例如：a * b ≠ b * a】
#   4.矩阵的基本理解
#   5.了解Sigmoid、tanh,ReLU等等激活函数
#   6.各种常用的倒数公式及复合函数求导
#课前补充知识：
#   1.torch.randn(x,y) = np.random.randn(x,y)
#   2.pytorch中的点积是：torch.mm(),用法:torch.mm(mat1, mat2, out=None) → Tensor
#对矩阵mat1和mat2进行相乘。 如果mat1 是一个n×m 张量，mat2 是一个 m×p 张量，将会输出一个 n×p 张量out。
#   3.pytorch中的RelU激活函数是torch.clamp(),用法：torch.clamp(input, min, max, out=None) → Tensor
#   4.Pytorch中的转置是torch.t(),用法：torch.t(input) → Tensor，关于x.item()用法及理解：
#文档中给了例子，说是一个元素张量可以用item得到元素值，请注意这里的print(x)和print(x.item())值是不一样的，
#一个是打印张量，一个是打印元素
#   5.clone() → Tensor:用法:返回与原tensor有相同大小和数据类型的tensor

'''
特别注意，该代码与Learning_torch_numpy.py中的代码无任何区别，都属于手动求导，只不过使用了tensor张量进行取值计算。
'''
#课前知识：
#   1.torch.tensor(,requires_grad = true)默认requires_grad = true,可以omit
#   2.自动求导：y.backward()——如果在张量中，允许求导之后，那么自动会在储存的张量中进行储存求导结果。
#   3.自动求导的理解：因为是对y进行求导，那么如果允许的话，可以直接输出对应求导的数字：
#例如： dy / dx ,那么输出为：x.grad()——>伪代码：对x的导数值
#这里还是涉及到了链式求导法则的相关知识，在torch中的backward()相关的使用中，不用进行一步一步的手动求导公式的推导，
#而是机器在处理张量的数据的时候，如果检测到requires_grad = true的话，那么在计算前向网络时自动进行反向传播求导的过程，
#然后我们可以直接的进行访问，啰嗦了，换句话讲的话，就是链式求导法则中，省略了中间步骤，直接进行相应倒数的输出。
#举个例子：y(f(g(x)))'=>可以直接对g(x)进行求导，伪代码：
#   y.backward()
#   g(x).grad
#   x.grad
#实际的完整的例子：
#   x = torch.tensor(1,requires_grad = True)
#   w = torch.tensor(2,requires_grad = True)
#   b = torch.tensor(3,requires_grad = True)
#   y = w*x + b
#   y.backward()
#   print(x.grad)——>输出：y' = w , print():1
#   print(w.grad)——>输出：y' = x , print():2
#   print(b.grad)——>输出：y' = 1 , print():1
#   print(y.grad)——>输出：1 , print():''【nil】
import torch


N,D_in,H,D_out=64,1000,100,10


x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)


learning_rate = 1e-6


for it in range(500):

    h = torch.mm(x,w1) #N * H

    h_relu = h.clamp(min=0) 
    y_pred = h_relu.mm(w2) 
    


    loss = (y_pred - y).pow(2).sum().item()
    print(it,loss)
    
    #Backwward pass 反向传播
    #compute the gradient 梯度求导(向后看)

    
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)

    
    #update weights of w1 and w2

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    

















































