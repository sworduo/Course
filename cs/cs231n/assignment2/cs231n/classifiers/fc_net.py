from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        #注释这么写会好很多，不过不注释的是我第一次写的，记录一下naive的写法吧
        #self.params['W1] = np.random.normal(0, weight_scale, size=(input_dim, hidden_dim)
        self.params['W1'] = np.random.normal(0, weight_scale, input_dim * hidden_dim).reshape(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, hidden_dim * num_classes).reshape(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # 也可以调用relu_forward和affine_forward完成
        relu_output = np.maximum(np.reshape(X, (X.shape[0], -1)).dot(self.params['W1'])+self.params['b1'], 0)
        scores = relu_output.dot(self.params['W2'])+self.params['b2']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # shift_scores = scores - np.max(scores, axis=1, keepdims=1)
        # sample_scores_sum = np.sum(np.exp(shift_scores), axis=1, keepdims=1)
        # log_scores = shift_scores - np.log(sample_scores_sum)
        # N = scores.shape[0]
        # loss = -np.sum(log_scores[np.arange(N), y]) / N
        # loss += 0.5 * self.reg * (np.sum(np.multiply(self.params['W1'], self.params['W1'])) + np.sum(np.multiply(self.params['W2'], self.params['W2'])))
        # probs = np.exp(log_scores)
        # probs[np.arange(N), y] -= 1.0
        # probs /= N
        # dW2 = relu_output.T.dot(probs)
        # d_relu_out = probs.dot(self.params['W2'].T)
        # db2 = np.sum(probs, axis=0)
        # d_relu_in = np.multiply((relu_output > 0)*1, d_relu_out)
        # dW1 = X.T.dot(d_relu_in)
        # db1 = np.sum(d_relu_in, axis=0)
        loss, d_softmax_input = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(np.multiply(self.params['W1'], self.params['W1'])) + np.sum(np.multiply(self.params['W2'], self.params['W2'])))
        d_relu_out, dW2, db2 = affine_backward(d_softmax_input, (relu_output, self.params['W2'], self.params['b2']))
        d_relu_in = relu_backward(d_relu_out, relu_output)
        dx, dW1, db1 = affine_backward(d_relu_in, (X, self.params['W1'], self.params['b1']))
        grads['W2'] = dW2 + self.reg * self.params['W2']
        grads['b2'] = db2
        grads['W1'] = dW1 + self.reg * self.params['W1']
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        hidden_dims.insert(0, input_dim)
        #如果一开始有C个隐藏层,那么最后应该有C+1个W
        #前C个W对应于将前一层数据映射到当前层(第一个W是将输入层映射到第一个隐藏层)
        #最后一个W对应于将最后一个隐藏层映射到输出层.
        #overall,L层隐藏层,意味着总共有L+1层,因此需要L+1个权重矩阵
        for i in range(1, len(hidden_dims)):
            input_num = hidden_dims[i-1]
            output_num = hidden_dims[i]
            self.params['W'+str(i)] = np.random.normal(0, weight_scale, input_num * output_num).reshape(input_num, output_num)
            self.params['b'+str(i)] = np.zeros(output_num)
            if self.use_batchnorm:
                self.params['gamma'+str(i)] = np.ones(output_num)
                self.params['beta'+str(i)] = np.zeros(output_num)
        self.params['W'+str(self.num_layers)] = np.random.normal(0, weight_scale, hidden_dims[-1] * num_classes)\
                                                .reshape(hidden_dims[-1], num_classes)
        self.params['b'+str(self.num_layers)] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # #layers_input/output从输入层,也就是说X开始算起,layers_input初始化为1是保持下标一致
        # #layers_input/output大小为self.num_layers,包括输入层到所有隐藏层的输入输出,不包括最后一层softmax的输入输出.
        # layers_input = [1]
        # #X可能是一个张量.
        # layers_output = [np.reshape(X, (X.shape[0], -1))]
        # #完成数据在L个隐藏层之间的传输
        # for i in range(1, self.num_layers):
        #     #affine
        #     affine_out = layers_output[-1].dot(self.params['W'+str(i)])+self.params['b'+str(i)]
        #     layers_input.append(affine_out)
        #     #relu
        #     scores = np.maximum(affine_out, 0)
        #     layers_output.append(scores)
        # #将数据从最后一个隐藏层映射到输出层
        # scores = layers_output[-1].dot(self.params['W'+str(self.num_layers)])+self.params['b'+str(self.num_layers)]
        #每一个affine-bn-relu的缓存
        caches = {}
        dropoutCache = {}
        # X可能是张量
        out = np.reshape(X, (X.shape[0], -1))
        for i in range(1, self.num_layers):
            if self.use_batchnorm:
                out, caches[i] = affine_bn_relu_forward(out, self.params['W'+str(i)], \
                                                                     self.params['b'+str(i)], self.params['gamma'+str(i)], \
                                                                     self.params['beta'+str(i)], self.bn_params[i-1])
            else:
                out, caches[i] = affine_relu_forward(out, self.params['W'+str(i)],\
                                                                  self.params['b'+str(i)])
            if self.use_dropout:
                out, dropoutCache[i] = dropout_forward(out, self.dropout_param)
        #最后一层输出
        scores, caches[self.num_layers] = affine_forward(out, self.params['W'+str(self.num_layers)], \
                                                                   self.params['b'+str(self.num_layers)])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #dout是softmax层输入的梯度
        loss, dout = softmax_loss(scores, y)
        #dout是最后一个隐藏层输出的梯度;
        # dout, grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = \
        #                                     affine_backward(dout, (layers_output[-1], \
        #                                                            self.params['W'+str(self.num_layers)], \
        #                                                            self.params['W'+str(self.num_layers)]))
        dout, grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = \
                                            affine_backward(dout, caches[self.num_layers])
        loss += 0.5 * self.reg * np.sum(self.params['W'+str(self.num_layers)] * self.params['W'+str(self.num_layers)])
        grads['W'+str(self.num_layers)] += self.reg * self.params['W'+str(self.num_layers)]
        #假设有L个隐藏层权重矩阵,1个最后一层到输出层的权重矩阵,共self.num_layers个权重矩阵
        #现在剩下L个权重矩阵需要更新
        for i in range(self.num_layers-1, 0, -1):
            loss += 0.5 * self.reg * np.sum(self.params['W'+str(i)] * self.params['W'+str(i)])
            if self.use_dropout:
                dout = dropout_backward(dout, dropoutCache[i])
            if self.use_batchnorm:
                dout, grads['W'+str(i)], grads['b'+str(i)], grads['gamma'+str(i)], grads['beta'+str(i)] \
                    = affine_bn_relu_backward(dout, caches[i])
            else:
                dout, grads['W'+str(i)], grads['b'+str(i)] = affine_relu_backward(dout, caches[i])
            grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
        # for i in range(self.num_layers, 1, -1):
        #     index = self.num_layers - i
        #     dout, grads['W'+str(index)], grads['b'+str(index)] = \
        #                                      affine_relu_backward(dout, \
        #                                       ((layers_output[index-1], \
        #                                                 self.params['W'+str(index)], \
        #                                                 self.params['b'+str(index)]),\
        #                                        layers_input[index]))
        #     grads['W'+str(index)] += self.reg * self.params['W'+str(index)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
