from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ########################################################################### 
    #假如x是一组图片,那么N就是图片数量,d_1是行,d_2是列,d_3是rgb分量
    #从affine_backward来看,似乎cache保存的是原来的x,而不是reshape后的x
    #x = np.reshape(x, (x.shape[0], -1))
    #out = x.dot(w)+b
    out = np.reshape(x, (x.shape[0], -1)).dot(w)+b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    xshape = x.shape
    xtmp = np.reshape(x, (xshape[0], -1))
    dw = xtmp.T.dot(dout)
    db = np.sum(dout, axis=0)
    dxtmp = dout.dot(w.T)
    dx = np.reshape(dxtmp, xshape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.multiply((x > 0) * 1, dout)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        # 用每次得到的mean和var迭代更新test需要用到的mean和test
        #而不是保存每次的mean和var然后无偏估计出test所要用到的mean和var
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        xhat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * xhat + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = (gamma, x, xhat, sample_mean, sample_var, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        out = gamma * (x - running_mean) / np.sqrt(running_var + eps) + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    gamma, x, xhat, mean, var, eps = cache
    dgamma =  np.sum(dout * xhat, axis=0)
    dbeta = np.sum(dout, axis=0)

    dxhat = dout * gamma
    N = x.shape[0]
    a = np.sqrt(var + eps)
    dvar =  np.sum((x - mean) * dxhat * (-0.5) / (a**3), axis=0)
    dmean = np.sum(- dxhat / a, axis=0) + dvar * np.sum(-2 * (x-mean), axis=0)
    dx = dxhat / a + dmean / N + dvar * 2 * (x - mean) /N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    gamma, x, xhat, mean, var, eps = cache

    dgamma = np.sum(dout * xhat, axis=0)
    dbeta = np.sum(dout, axis=0)

    N = x.shape[0]
    a = np.sqrt(var + eps)
    dxhat = dout * gamma
    dvar = np.sum((x - mean) * dxhat * -0.5 / a**3, axis=0)
    dmean = np.sum( - dxhat / a, axis=0)  #+dvar * np.sum(-2 * (x - mean), axis=0) /后面这串近似=0
    dx = dxhat / a + dvar * 2 * (x - mean) / N + dmean / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    newH = 1 + (H + 2 * pad - HH) // stride
    newW = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, newH, newW))
    #zero_pad
    padX = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    #以滤波器左上角为起始点进行考虑，而不要以滤波器中心为起始点进行考虑。
    for hindex in range(newH):
        for windex in range(newW):
            #stride
            maskX = padX[:, :, hindex*stride:hindex*stride+HH, windex*stride:windex*stride+WW]
            for thisFilter in range(F):
                #axis=0是所有数据，1是通道，2是h，3是w
                out[:, thisFilter, hindex, windex] = np.sum(maskX * w[thisFilter], axis=(1,2,3))
    #填充None就是扩张b的维度。
    #[1,2,3,4] 3*4维合起来是一张图片，每一个第2维对应F张图片
    #假如b=[1,2,3]
    #然后b[None, :, None, None]意味着将b的为None的维度扩张的和其他参数（out）一样，然后保持:的维度是本身的数字。
    #假设扩张之后的维度是b.shape=(2,3,4,5)
    #那么此时的具体含义是，有两个3维数组，每个三维数组里面有3个二维数组，这3个二维数组分别维全1，全2，全3.
    #又比如说，如果b[:,None,None,None]，令此时b.shape=(3,4,5,6)
    #代表有3个三维数组，每个三维数组分别为全1，全2，全3
    #又比如b[None,None,:,None],b.shape=(2,4,3,5)
    #代表有2个三维数组，每个三维数组里有4个二维数组，每个二维数组有3个大小为5一维数组，其值分别为全1全2全3.
    #自己实验一下就明白了。
    out += b[None, :, None, None]

    #------------------------------------------------------------
    #多重循环，too naive实现
    #而且下面的代码，在考虑卷积的时候，是以卷积核中心，然后计算图像中心对应的左边界和右边界，然后再与卷积核对应相乘
    #这么写很复杂，第一，要考虑卷积核大小是奇数还是偶数，因为奇偶数会影响右边界
    #第二这样要考虑起始点和计算终止点
    #事实上不用这么复杂，因为卷积核是从图像左上角开始扫的，不要从卷积核中心角度去思考，而是要从整个卷积核本身去思考。
    #比如，从卷积核中心角度去思考的时候，我需要计算picture[cur-h/2:cur+h/2+1]，这里右边界会因为滤波器奇偶不同而不同。
    #而以卷积核本身去思考的时候，我只要从图像左上角开始picture[cur:cur+h]就可以，根本不用考虑滤波器是奇数还是偶数。
    # padH = H + 2 * pad
    # padW = W + 2 * pad
    # #下标从0开始
    # halfH = HH // 2
    # halfW = WW // 2
    # #滤波器大小可能是偶数，所以右边边界需要仔细考虑
    # #对于大小为偶数的滤波器而言，比如一个4*4的滤波器，其下标是0,1,2,3，那么中心在2处。
    # #以下的代码是基于上面的假设的，并且能通过测试。
    # HrightBound = halfH + (0 if ((HH % 2) == 0) else 1)
    # WrightBound = halfW + (0 if ((WW % 2) == 0) else 1)
    # #range左闭右开
    # for thisData in range(N):
    #     pic = np.pad(x[thisData], ((0,0), (pad, pad), (pad, pad)), 'constant')
    #     for thisFilter in range(F):
    #         i = 0
    #         for thisH in range (halfH, padH - HrightBound + 1, stride):
    #             j = 0
    #             for thisW in range (halfW, padW - WrightBound + 1, stride):
    #                 out[thisData, thisFilter, i, j] = np.sum(w[thisFilter] * \
    #                                                          pic[:, thisH - halfH:thisH + HrightBound, \
    #                                                          thisW - halfW:thisW + WrightBound]) + b[thisFilter]
    #                 j += 1
    #             i += 1
    # ------------------------------------------------------------
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    #假设前向传播的时候X=(N,C,H,W) W=(F,C,HH,WW)
    #那么dout的大小应该是dout=(N,F,newH,newW) newH=1+(H+2*pad-HH)//stride
    #具体到dout[数据d，滤波器f，h,w]这一点，这一点相当于是原来的w[f,C,HH,WW]和数据d在（h，w）处的仿射变换
    #即是dout[d,f,h,w] = w[f,C,HH,WW] * Xpad[d,C,h,w] （Xpad就是zero-pad之后的图像)
    #看着是不是很熟悉，就是affine阿！具体到每个w[f,c,hh,ww]的导数就是dout[n,f,h,w]*Xpad[d,c,h,w]接下来就好办了。

    #关于求导，只要找到x对哪些项有贡献就行了。如果x对多个项有贡献，那就将多个项结合起来就行了。
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad, stride = conv_param['pad'], conv_param['stride']
    newH, newW = dout[0,0].shape
    xpad = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    dw = np.zeros_like(w)
    dxpad = np.zeros_like(xpad)

    db = np.sum(dout, axis=(0, 2, 3))
    for hindex in range(newH):
        for windex in range(newW):
            xpad_mask = xpad[:, :, hindex*stride:hindex*stride+HH, windex*stride:windex*stride+WW]
            #对于输出图像(newH, newW)的每个点y，y=x11*w11+...+x33*w99 + b（假设是3*3滤波器）
            #dy每个点对x和w的导数都是类似的，不同的是，w是对所有数据都有贡献，而某个数据的x只对对应生成的滤波图像有贡献
            #而b，是对每个数据生成的每个滤波图像都有贡献，所以w对n求和，x对F求和，b对n和F都求和。
            for fi in range(F):
                dw[fi, :, :, :] += np.sum(xpad_mask * dout[:, fi, hindex, windex][:,None,None,None], axis=0)
            for ind in range(N):
                dxpad[ind, :, hindex*stride:hindex*stride+HH, windex*stride:windex*stride+WW] += \
                np.sum(dout[ind, :, hindex, windex][:,None,None,None] * w, axis=0)
    dx = dxpad[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    #每个数据都是三维数组，可以看成是，每个数据都是由多个滤波图像组成的图像簇
    N, C, H, W = x.shape
    poolH = pool_param.get('pool_height', 2)
    poolW = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 1)
    newH = (H - poolH) // stride + 1
    newW = (W - poolW) // stride + 1
    out = np.zeros((N, C, newH, newW))
    for hindex in range(newH):
        for windex in range(newW):
            out[:, :, hindex, windex] = np.max(x[:, :, hindex*stride:hindex*stride+poolH, windex*stride:windex*stride+poolW], axis=(2,3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    #平均池化求导，假设输出y = (x1+x2+x3+x3)/4
    #显然每一个x的导数都是 dy * 1/4,因为y对x1的导数就是1/4
    #最大池化，则是region内最大值有梯度，其余无梯度
    #dout的维度：N-多少组图片（数据大小），C-一组图片有多少张（多少个滤波器得到的），newH,newW
    x, pool_param = cache
    poolH, poolW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W = x.shape
    newH = (H - poolH) // stride + 1
    newW = (W - poolW) // stride + 1
    dx = np.zeros_like(x)
    for hindex in range(newH):
        for windex in range(newW):
            xmask = x[:, :, hindex*stride:hindex*stride+poolH, windex*stride:windex*stride+poolW]
            #find the index of the max value
            #the mask value corresponding to the max value inside the pool region is one, the rest is zero
            maxValMask = (xmask == np.max(xmask, axis=(2,3))[:, :, None, None])
            dx[:, :, hindex*stride:hindex*stride+poolH, windex*stride:windex*stride+poolW] += \
                    dout[:, :, hindex, windex][:, :, None, None] * maxValMask

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape
    out, cache = batchnorm_forward(np.reshape(x.transpose(0,2,3,1), (-1, C)), gamma, beta, bn_param)
    out = np.reshape(out, (N, H, W, C)).transpose(0,3,1,2)#注意这里reshpe的参数
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dx, dgamma, dbeta = batchnorm_backward(dout.transpose(0,2,3,1).reshape(-1, C), cache)
    dx = dx.reshape(N, H, W, C).transpose(0,3,1,2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
