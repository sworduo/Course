import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
    scores = np.exp(X[i].dot(W))
    probability = scores/np.sum(scores)
    loss -= np.log(probability[y[i]])
    dW[:, y[i]] += -X[i]
    for j in range(num_class):
      dW[:, j] += probability[j]*X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.exp(X.dot(W))
  row_sum = np.sum(scores, axis=1, keepdims=True)
  probability = scores/row_sum

  loss = -np.sum(np.log(probability[np.arange(num_train), y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  probability[np.arange(num_train), y] -= 1.0
  dW = np.dot(X.T, probability)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

