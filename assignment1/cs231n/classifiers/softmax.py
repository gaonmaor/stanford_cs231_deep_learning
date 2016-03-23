import numpy as np
from random import shuffle


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
    num_classes = W.shape[1]
    num_train, dim_size = X.shape
    for i in xrange(num_train):
        scores = X[i, :].dot(W)
        scores -= np.max(scores)
        exp_sum = 0.0
        for j in xrange(num_classes):
            exp_sum += np.exp(scores[j])
        for j in xrange(num_classes):
            dW[:, j] += np.exp(scores[j]) * X[i, :] / exp_sum
            if j == y[i]:
                dW[:, j] -= X[i, :]
        loss += -scores[y[i]] + np.log(exp_sum)

    # Average gradients as well
    dW /= num_train

    # Add regularization to the gradient
    dW += reg * W

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train, dim_size = X.shape
    scores = X.dot(W)
    scores -= np.max(scores, axis = 1).reshape(num_train, 1)
    correct_scores = scores[range(num_train), y]
    exp_score = np.exp(scores)
    exp_sum = np.sum(exp_score, axis = 1)
    loss = np.sum(-correct_scores + np.log(exp_sum))
    probs = exp_score / exp_sum.reshape(-1, 1)
    probs[range(num_train), y] -= 1
    dW = X.T.dot(probs)

    # Average gradients as well
    dW /= num_train

    # Add regularization to the gradient
    dW += reg * W

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW