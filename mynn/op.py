from abc import abstractmethod

import numpy as np


class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # Initialize weights and biases
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))  # Bias initialized to zeros
        self.grads = {'W': None, 'b': None}
        self.input = None  # To store input for backward pass

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay  # Whether weight decay is applied
        self.weight_decay_lambda = weight_decay_lambda  # Control the intensity of weight decay

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        output: [batch_size, out_dim]
        """
        self.input = X  # Save the input for the backward pass
        # Compute the linear transformation
        output = np.dot(X, self.W) + self.b
        return output

    def backward(self, grad: np.ndarray):
        """
        input: grad: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        Also calculates the grads for W and b.
        """
        # Compute the gradient w.r.t W, b, and X
        batch_size = grad.shape[0]
        
        # Gradient with respect to W
        self.grads['W'] = np.dot(self.input.T, grad)  # X.T * grad
        
        # Gradient with respect to b
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)  # Sum over batch
        
        # Gradient with respect to the input X (for backpropagation)
        grad_input = np.dot(grad, self.W.T)  # grad * W.T
        
        # If weight decay is enabled, add the regularization term
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W  # Regularization on weights
        
        return grad_input

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        pass

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        pass

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax.
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss using softmax and cross-entropy.
        """
        if self.has_softmax:
            predicts = softmax(predicts)
        
        # Avoid log(0) by adding a small epsilon value
        epsilon = 1e-10
        predicts = np.clip(predicts, epsilon, 1.0 - epsilon)  # Ensure no values are exactly 0 or 1
        
        batch_size = predicts.shape[0]
        log_probs = np.log(predicts[np.arange(batch_size), labels])
        
        # Calculate the loss as the negative log likelihood
        loss = -np.mean(log_probs)
        
        # Backpropagate gradients
        self.grads = predicts
        self.grads[np.arange(batch_size), labels] -= 1
        self.grads /= batch_size
        
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        self.model.backward(self.grads)
        
    def cancel_softmax(self):
        self.has_softmax = False
        return self

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)


    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition