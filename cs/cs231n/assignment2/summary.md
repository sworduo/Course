#	梯度检查
看梯度检查源码时,要先注意一下传进去的f返回值是一个数,还是一个向量,又或者是一个矩阵.必须先弄明白这个,不然很容易看得懵逼.  
#	affine_forward/backward
这是计算这层输出,和下层输入的,可以看成是一个全连阶层的前向传播和反向传播对.   
#	softmax
参考代码里实现的softmax不需要除法!  
#	optimal
将梯度下降,前反向传播,训练过程分割开,源代码值得一读.  
#	梯度更新算法
一种是优化学习速率,一种是优化学习方向,adam是结合两种.  
#	bn
*	bn计算test用到的mean和var有两种方法,一种是保存每个minibatch计算得到的mean和var,然后无偏估计出test要用到的mean和var;另一种方法是,每个minibatch计算得到mean和var之后,使用动量更新方法,迭代求出最后test要用到的mean和var.这里用到的是后一种方法.  
*	gamma等于改变方差,反映到分布上,就是拉伸分布.beta的等于改变均值,反映到分布上就是移位.  
*	BN对x求导要注意!!这是个坑啊,因为m和var都有x,所以对x求导等价于对(x-m)/(含x的表达式)求导,展开后一共是三项,对x求导+对m再对x求导+对var再对x求导. 其中还有一个坑,var里面也是含有m的!对m求导要分成两部分,可以看成是对(x-m) * (1/(含m的式子)) 求导.所以求导起来比较麻烦.所以总结起来就是d(x-m)/(sig2+?) /dx + d(x-m)/(sig2+?)/dm + d(x-m)/(sig2+?)/d(sig2),后面两项可以看成是(dxhat/dm) * (dm/dx) 以此类推.后面写成公式的形式好一点.   
*	对于同一个minibatch,所有xhat对同一特征的不同sample的均值/方差的导数应该是一样的.  
*	优化的BN算法，我也不知道哪里可以优化- -，参考了下别人的代码，似乎是某个地方太小舍去了，因而增加了速度（并没有增加多少,而且增加的速度不稳定）。  

#	模块化
*	模块化编程的好处在于我们只需要关心每个模块之间的关系，而不需要关心模块内部的数据。  
比如将bn的forward和backward写好，甚至用affine_bn_relu将三个层拼接好之后，我们可以把此时affine_bn_relu三个曾当成是一个层来思考，相应的前/反向传播由写好的的函数来完成，这里反向传播所需要用到的变量用cache来保存就是了，这样我们就不用瞎操心每次反向传播应该传什么参数进去，可以把我们的注意力思考在模块之间的拼接。而且可以使代码可读性更高。  

#	bn错误
*	在训练bn的时候,发现很奇怪,最后一层的w在reg=0的时候没有错,reg不等于0的时候有错,而其他所有的w,b,gamma,beta都没有错.一开始我以为是最后一层w没有加正则项的原因,后来发现加了.仔细检查代码,发现原来是是最后的loss忘了加最后一层w的正则项了,所以验证梯度的时候,其余的w正则项都加进loss了,所以没有错,而最后一层w正则项忘了加进loss,表现出来就是用不加正则项的loss计算出来的w会偏大一点,相当于是一个对dloss=d(wx+b)求导,一个是对dloss=d(wx+b+0.5*reg*w*w)结果当然会出现偏差了.那么为什么其他w没错呢?因为其他w的正则项都在loss里,求导公式是一样的.  

#	dropout
*	假设输入是x，那么经过dropout之后，输出是y = (px + (1-p)\*0)=px，此时为了保证预测时候和训练时候数据的一致，预测时前向传播也需要乘以p。因为我们训练的时候是按px来训练的，预测的时候如果不乘以p，那么预测时候网络流过的数据就是x，和我们训练时候的模型是不一样的，所以需要处以p。一般而言，为了保证不管是否使用dropout，预测代码都是一样的，所以一般情况下，我们都会在使用dropout时，直接在训练时除以p，也就是说，训练时y = (px+(1-p)\*0)/p=x，这样就保证训练的时候也是按输入是x的模型来训练了。然而这里其实也有点小细节需要注意，比如一开始我的代码是这么写的：

```python
#错误代码
def dropout_forward(x, p)
	mask = np.random.rand(*x.shape) < p
	out = x * mask
	out = out / p
	cache = mask
	return out, cache

def dropout_backward(dout, cache)
	mask = cache
	dx = dout * cache
```

*	上面的代码前向传播是对的，然而反向传播是错误的。因为和前向传播一样，假设drouput层进来的梯度是dout，同样的dx = p\*dout + (1-p)\*0 = p\*dout,这是不对的，因为前向传播是按x来转移数据，而此时反向传播却是按p\*dout来更新矩阵，所以出了偏差。正确的做法是dx同样除以p才行。事实上，更好的做法是在生成mask的时候就除以p，这样就可以少一次除法了，当x非常大并且都是浮点数的时候，少一次除法的吸引力是非常大的。

```python
#正确代码
def dropout_forward(x, p)
	mask = (np.random.rand(*x.shape) < p) / p
	out = x * mask
	return out, mask

def dropout_backward(dout, cache)
	mask = cache
	dx = dout * mask
	return dx
```

*	spatial batch normalization
batch normalize可以看成是经过仿射变换之后，对放射变换的输出，下一层神经元的输入进行数据归一化。可以看成是对**同一个**仿射变换单元输出的数据进行归一化。  
当BN技术应用于CNN卷积层时，可以看成对是卷积层的输出，下一层神经元（一般是relu）的输入进行数据归一化。也可以看成是**同一个**卷积核输出的像素点的归一化。假设一开始输入的图像是(N,C,H,W)，C是维数，N是数据大小，经过卷积层之后，变成（N，F，H，W），F是卷积核的数量。这里我们把一个卷积核看成是一个神经元，相当于是参数共享（不然要学习的参数太多了）。那么我们思考一下，每个卷积核的输出数据有哪些，或者说，输出图像的哪些像素点与卷积核有关系。事实上，cnn输出的每一组对应于卷积核f的图像的每个像素点都和卷积核f有关系！也就是说对应卷积核f的数据是所有数据对应卷积核f的滤波图像的所有像素点的并集，此时需要对其所有数据的所有图像像素点进行归一化，具体来说就是对(:,f,:,:)(axis=0,2,3)做归一化。这时候其实我们只要将卷积层输出transpose一下，然后reshape，使得每一列都对应于一组不同滤波图像的一组值，此时同一列的元素对应于同一个卷积核产生的所有像素点，简单来说此时是(N\*H\*W, F)。代码如下:

```python
N, F, H, W = x.shape
#batchnorm_forward是给放射变换写好的bn，具体可以参考cs231n assignment2的大作业
out, cache = batchnorm_forward(x.transpose(0,2,3,1).reshape(-1,F), gamma, beta, bn_params)
out = out.reshape(N, H, W, F).transpose(0, 3, 1, 2)#注意这里reshape的参数
```

*	关于transpose,假设四维张量（3，4，5，6）是3组数据，每组数据有4张滤波图片，每张图片大小是5，6正常的时候，是按顺序，第一组第一张图片第一行数据遍历，然后是第一组第一张图片第二行...  
如果应用了x.transpose(0,1,3,2)，此时x.shape=(3,4,6,5)，可以看成是对第一组第一张图片的第一列遍历，然后是第二列，第三列...  
如果应用了x.transpose(0,2,3,1)，此时x.shape=(3,5,6,4)，可以看成是对第一组**所有图片**的第一个像素进行遍历，也就是说，这时候的一项相当于transpose前每一组**所有滤波图片**对应像素点组成的列表。  
简而言之，假设水平向右是x轴，水平向下是y轴，垂直向下是z轴，则transpose(0,1,3,2)是按y轴遍历，transpose(0,2,3,1)是按z轴遍历。  


