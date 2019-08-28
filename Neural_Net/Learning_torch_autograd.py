# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:08:27 2019

@author: test

本次使用Pytorch 创建向前神经网络，计算损失以及反向求导。
在本次的代码中，使用的是：autograd，自动求导的功能。
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
#   6.torch.from_numpy(x),改用法是将numpy中创建的数组转成tensor的格式。
'''
特别注意，改代码实现了自动求导，所以不用进行一步一步的求导，建议对比后进行学习。
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

w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6

for it in range(500):

    #Forward pass
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)
    
    #compute loss
    loss = (y_pred - y).pow(2).sum()
    print(it,loss.item())
    
    #Backwward pass 反向传播
    #compute the gradient 梯度求导(向后看)
    loss.backward()
    
    #update weights of w1 and w2
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
#补充讲解Python关键字：（可能是血的太不牢靠了，with用法都忘光了）    
#   1.所以使用with处理的对象必须有__enter__()和__exit__()这两个方法。
#   2.with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，
#释放资源，比如文件使用后自动关闭、线程中锁的自动获取和释放等。
#   3.with基本用法：
#    with expression [as target]:
#       with_body
#   4.一般在操作文件时，会进行操作，因为存在自动关闭的机制，所以省略了：file.close()操作。
#   5.如果注释掉with torch.no_grad():，那么将会出现下面的错误，
#       RuntimeError:a leaf Variable that requires grad has been used in an in-place operation.
        #翻译：就地操作中已使用了需要渐变的叶变量。
#   出现这个错误的原因是：因为我们在已经在内存中分配了相关的空间储存导数的模型，所以，不能进行叠加，那么
#   我们就通过这个操作来进行。
        #可以参照博客：http://blog.sina.com.cn/s/blog_a99f842a0102y1e4.html
#       #Pytorch入门参考博客：https://blog.csdn.net/u014380165/article/details/78525273【入门参考】
#       #Pytorch源码解读：https://blog.csdn.net/u014380165/column/info/19413【源码解读】










    
    
    
    
    
    
    
    
