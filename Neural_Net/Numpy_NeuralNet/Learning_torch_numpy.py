# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:08:51 2019

@author: test

目标：神经网络学习
基本的pytorch学习与numpy学习
热身：使用numpy实现两层神经网络
一个全连接ReLU神经网络，一个隐藏层，没有bias(偏移量),用来从x预测y,使用L2 Loss.
h=w_1*x+b_1——>第一层
a=max(0,h)——>ReLU层
y_hat=w_2*a+b_2——>第二层
在本次的神经网络是取消了bias，只是理解神经网络的基本构成
    *forward pass 前向传播
    *loss 损失值
    *backward pass 反向传播 
这一实现完全使用numpy来计算前向神经网络，loss和反向传播.
numpy ndaray是一个普通的n维array,他不知任何关于深度学习或者是梯度(gradenit的知识),
也不知道计算图(computation graph),他只是一种用来计算数学运算的数据结构
"""
#这里需要很多数学的知识
#   1.导数基础——链式求导法则
#   2.导数的意义——值的含义，就是每个值代表的什么
#   3.矩阵的运算基础，内积(点积)【这里需要注意的问题是矩阵的点积是不能进行前后变幻的，就例如：a * b ≠ b * a】
#   4.矩阵的基本理解
#   5.了解Sigmoid、tanh等等激活函数
#这些都需要之后进行补充
import numpy as np

#N：输入的数量 （训练数据的量） 
#D_in：输入层的维度
#H：隐藏层的维度
#D_out：输出的数量，也可以成为维度
N,D_in,H,D_out=64,1000,100,10

#创建一些训练数据
#在这里创建数据的时候，我们可以看到维度
#x——>创建了N行D_in列
#y——>创建了N行D_out列
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

#w_1是将输入从1000维转成100的向量，就是从输入层到第一层的变换（利用向量的点积进行降维处理）
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

#学习效率
learning_rate = 1e-6

#这里的500就是训练的迭代次数
#epoch的含义在于此，也就是相当于一组数据训练500次
for it in range(500):
    #Forward Pass——>前向传播
    h = x.dot(w1) #N * H
    #这里的激活函数是：h与0进行比较取最大值
    #激活函数存在的意义：解决线性不可分的问题，具体的剖析，参考：
    #https://blog.csdn.net/program_developer/article/details/78704224
    #总结：总结：激活函数是用来加入非线性因素的，提高神经网络对模型的表达能力，解决线性模型所不能解决的问题。
    h_relu = np.maximum(h, 0) #激活函数 N * H（激活函数维度不改变）
    y_pred = h_relu.dot(w2) #输出y N * D_out
    

    #compute loss——>计算损失值（这里使用均方误差损失值计算）
    #这里使用sum()的原因是因为，想要计算整个的矩阵的每一个元素的损失值之和
    #矩阵的加减法，加减法是针对对应元素的加减，最后得到的是一个数，而不是一个向量。
    loss = np.square(y_pred - y).sum()
    print(it,loss)
    
    #Backwward pass 反向传播
    #compute the gradient 梯度求导(向后看)
    # Example:   y = ax + b
    # 求导法则： dy / dx = a
    #           dy / da = x
    # 因为我们想要看一下，d(loss) / d(w1)，所以我们需要知道链式求导法则
    #所谓的链式求导法则就是：
    #   求 dy / dx
    #   那么我们就求 (dy / dt) * (dt / dx)——>这个公式在小学就学过了，经过约分
    #   其结果与(dy / dx)一样，只不过是将求导过程分解成小型的求导，下面的过程就是分解过程
    
    grad_y_pred = 2.0 * (y_pred - y)#因为上面额loss的值是(y_pred - y)^2
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)
    
    #update weights of w1 and w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    



