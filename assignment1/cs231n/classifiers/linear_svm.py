from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg, debug=False):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] # Num of classes (C) represents length of ['cat', 'dog', 'frog', 'car'...]
    num_train = X.shape[0] # Num of training examples (N) represents length of X
    loss = 0.0 # Initialize loss to 0
    
    for i in range(num_train): # Loop through all training examples
        
        scores = X[i].dot(W) # Take training example i and dot product with weights to get scores
        correct_class_score = scores[y[i]] # Get the score of the correct class by looking at the index of y[i] (real label) in scores

        if i <= 2 and debug: # Debugging
            print(f'\n-----\n{scores=}\n{scores.shape=}')
            print(f"{y[i]=}")
            print(f'{correct_class_score=}\n\n')
        
        for j in range(num_classes): # Loop through all classes
            
            if j == y[i]: # Skip the correct class
                continue # Skip to the next iteration

            margin = scores[j] - correct_class_score + 1 # This calculates the margin (i.e. the difference between the score of the correct class and the score of the incorrect class)

            if i <= 2 and debug: # Debugging
                print(f'{margin=}')

            if margin > 0: # If the margin is greater than 0, then we have a loss
                loss += margin # Add the margin to the loss
                dW[:, j] += X[i].T # Add the training example to the gradient of the incorrect class
                dW[:, y[i]] -= X[i].T # Subtract the training example from the gradient of the correct class

                if i <= 2 and debug: # Debugging
                    print(f"{loss=}")
                    print(f'{dW[:, j][:3]=}\n{dW[:, j].shape=}')
                    print(f'{dW[:, y[i]][:3]=}\n{dW[:, y[i]].shape=}\n')

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    if debug:
        print(f'---\n\nPRE REG: {loss=}')

    # Add regularization to the loss.
    loss += reg * np.sum(W * W) # Regularization is added to the loss by summing the square of the weights and multiplying by the regularization strength
    
    if debug:
        print(f'POST REG: {loss=}')

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += 2 * reg * W # Add the regularization gradient to the gradient

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg, debug_loss=False, debug_grad=False):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0] # Num of training examples (N) represents length of X
    scores = X.dot(W) # Dot product of training examples and weights to get scores
    correct_class_scores = scores[np.arange(num_train), y] # Get the scores of the correct classes by looking at the index of y (real labels) in scores
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1) # Calculate the margins (i.e. the difference between the score of the correct class and the score of the incorrect class)

    if debug_loss:
        print(f'\n{num_train=}')
        print(f'\n{scores[:2, :]=}\n{scores.shape=}')
        print(f'\n{correct_class_scores[:2]=}\n{correct_class_scores.shape=}')
        print(f'\n\nPRE-ZERO: {margins[:2, :]=}\n{margins.shape=}')

    margins[np.arange(num_train), y] = 0 # Set the margins of the correct classes to 0
    loss = np.sum(margins) / num_train # Calculate the loss by summing the margins and dividing by the number of training examples

    if debug_loss:
        print(f'\nPOST-ZERO: {margins[:2, :]=}\n{margins.shape=}\n')
        print(f'\nPRE REG: {loss=}')

    loss += reg * np.sum(W * W) # Add regularization to the loss

    if debug_loss:
        print(f'POST REG: {loss=}\n')

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    binary = margins # Create a binary matrix where the margins are greater than 0
    binary[margins > 0] = 1 # Set the binary matrix to 1 where the margins are greater than 0

    if debug_grad:
        print(f'\nPRE ROW-SUM: {binary[:2, :]=}\n{binary.shape=}\n')

    row_sum = np.sum(binary, axis=1) # Sum the rows of the binary matrix
    binary[np.arange(num_train), y] = -row_sum # Subtract the sum of the rows from the correct classes

    if debug_grad:
        print(f'\n{row_sum[:10]=}\n{row_sum.shape=}\n')
        print(f'\nPOST ROW-SUM: {binary[:2, :]=}\n{binary.shape=}\n')

    dW = X.T.dot(binary) / num_train # Multiply the training examples by the binary matrix and divide by the number of training examples
    dW += 2 * reg * W # Add regularization to the gradient

    if debug_grad:
        print(f'\n{dW[:2, :]=}\n{dW.shape=}\n')

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
