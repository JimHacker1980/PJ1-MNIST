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
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W': None, 'b': None}
        self.input = None  # Record the input for backward process.

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay  # Whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda  # Control the intensity of weight decay

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        output: [batch_size, out_dim]
        """
        # Store the input for backward pass
        self.input = X
        # Compute the output: Y = X @ W + b
        out = np.dot(X, self.W) + self.b
        return out

    def backward(self, grad: np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # Calculate gradients for W and b
        batch_size = self.input.shape[0]

        # Gradient of W: dL/dW = X.T @ grad / batch_size
        self.grads['W'] = np.dot(self.input.T, grad) / batch_size

        # Gradient of b: dL/db = sum(grad, axis=0) / batch_size
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size

        # If weight decay is enabled, add L2 regularization term
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        # Gradient to be passed to the previous layer: dL/dX = grad @ W.T
        grad_input = np.dot(grad, self.W.T)

        return grad_input

    def clear_grad(self):
        """
        Clear the gradients stored in self.grads.
        """
        self.grads = {'W': None, 'b': None}

class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_decay=False, weight_decay_lambda=1e-8):
        """
        2D Convolution Layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the convolutional kernel (square assumed)
        :param stride: Stride of the convolution
        :param padding: Number of zero padding added to all sides of input
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        # Initialize weights (Kaiming Normal) and bias
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / in_channels)
        self.b = np.zeros((out_channels, 1, 1))

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dX = None

        self.db = self.db.astype(np.float32)
        self.dW = self.dW.astype(np.float32)

        # Store input for backprop
        self.X = None
        self.optimizable = True
        self.params = {'W': self.W, 'b': self.b}

        # Store gradients in the 'grads' dictionary for compatibility with optimizers
        self.grads = {'W': self.dW, 'b': self.db}

    def forward(self, X):
        """
        Forward pass of Conv2D.
        :param X: Input tensor of shape (batch_size, in_channels, height, width)
        :return: Output tensor of shape (batch_size, out_channels, new_height, new_width)
        """
        batch_size, in_channels, H, W = X.shape
        assert in_channels == self.in_channels, "Input channels must match layer's in_channels"

        # Compute output dimensions
        new_H = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        new_W = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Apply padding
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_padded = X

        # Store input for backward pass
        self.X = X_padded

        # Initialize output
        output = np.zeros((batch_size, self.out_channels, new_H, new_W))

        # Perform convolution
        for i in range(new_H):
            for j in range(new_W):
                h_start, h_end = i * self.stride, i * self.stride + self.kernel_size
                w_start, w_end = j * self.stride, j * self.stride + self.kernel_size
                receptive_field = X_padded[:, :, h_start:h_end, w_start:w_end]  # Shape: (batch, in_channels, kernel_size, kernel_size)

                # Perform convolution using einsum (efficient matrix multiplication)
                output[:, :, i, j] = np.einsum('bchw,ochw->bo', receptive_field, self.W) + self.b.reshape(-1)

        return output

    def backward(self, d_out):
        """
        Backward pass of Conv2D.
        :param d_out: Gradient of loss with respect to output, shape (batch_size, out_channels, new_H, new_W)
        :return: Gradient with respect to input, shape (batch_size, in_channels, H, W)
        """
        batch_size, out_channels, new_H, new_W = d_out.shape
        _, in_channels, H_padded, W_padded = self.X.shape

        # Initialize gradients
        self.dW.fill(0)
        self.db.fill(0)
        dX_padded = np.zeros_like(self.X)

        # Compute gradients
        for i in range(new_H):
            for j in range(new_W):
                h_start, h_end = i * self.stride, i * self.stride + self.kernel_size
                w_start, w_end = j * self.stride, j * self.stride + self.kernel_size

                # Extract input slice
                receptive_field = self.X[:, :, h_start:h_end, w_start:w_end]

                # Compute gradient w.r.t. weights
                self.dW += np.einsum('bchw,bo->ochw', receptive_field, d_out[:, :, i, j])

                # Compute gradient w.r.t. input
                dX_padded[:, :, h_start:h_end, w_start:w_end] += np.einsum('bo,ochw->bchw', d_out[:, :, i, j], self.W)

        # Compute bias gradient
        self.db = np.sum(d_out, axis=(0, 2, 3), keepdims=True)

        # Remove padding from dX
        if self.padding > 0:
            self.dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            self.dX = dX_padded

        # Store gradients in 'grads' dictionary for optimizer
        self.grads['W'] = self.dW
        self.grads['b'] = self.db

        self.dX = self.dX.astype(np.float32)

        return self.dX

    def __call__(self, X):
        X = X.astype(np.float32)
        return self.forward(X)

# class conv2D(Layer):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,weight_decay=False,weight_decay_lambda=1e-8):
#         """
#         2D Convolution Layer
#         :param in_channels: Number of input channels
#         :param out_channels: Number of output channels
#         :param kernel_size: Size of the convolutional kernel (square assumed)
#         :param stride: Stride of the convolution
#         :param padding: Number of zero padding added to all sides of input
#         """
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.weight_decay=weight_decay
#         self.weight_decay_lambda=weight_decay_lambda

#         # Initialize weights (Kaiming Normal) and bias
#         self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / in_channels)
#         self.b = np.zeros((out_channels, 1, 1))

#         # Gradients
#         self.dW = np.zeros_like(self.W)
#         self.db = np.zeros_like(self.b)
#         self.dX = None

#         self.db.astype(np.float32)
#         self.dW.astype(np.float32)

#         # Store input for backprop
#         self.X = None
#         self.optimizable = True
#         self.params = {'W': self.W, 'b': self.b}

#         # Store gradients in the 'grads' dictionary for compatibility with optimizers
#         self.grads = {'W': self.dW, 'b': self.db}

#     def forward(self, X):
#         """
#         Forward pass of Conv2D.
#         :param X: Input tensor of shape (batch_size, in_channels, height, width)
#         :return: Output tensor of shape (batch_size, out_channels, new_height, new_width)
#         """
#         batch_size, in_channels, H, W = X.shape
#         assert in_channels == self.in_channels, "Input channels must match layer's in_channels"

#         # Compute output dimensions
#         new_H = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
#         new_W = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

#         # Apply padding
#         if self.padding > 0:
#             X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
#         else:
#             X_padded = X

#         # Store input for backward pass
#         self.X = X_padded

#         # Initialize output
#         output = np.zeros((batch_size, self.out_channels, new_H, new_W))

#         # Perform convolution
#         for i in range(new_H):
#             for j in range(new_W):
#                 h_start, h_end = i * self.stride, i * self.stride + self.kernel_size
#                 w_start, w_end = j * self.stride, j * self.stride + self.kernel_size
#                 receptive_field = X_padded[:, :, h_start:h_end, w_start:w_end]  # Shape: (batch, in_channels, kernel_size, kernel_size)

#                 # Perform convolution using einsum (efficient matrix multiplication)
#                 output[:, :, i, j] = np.einsum('bchw,ochw->bo', receptive_field, self.W) + self.b.reshape(-1)

#         return output

#     def backward(self, d_out):
#         """
#         Backward pass of Conv2D.
#         :param d_out: Gradient of loss with respect to output, shape (batch_size, out_channels, new_H, new_W)
#         :return: Gradient with respect to input, shape (batch_size, in_channels, H, W)
#         """
#         batch_size, out_channels, new_H, new_W = d_out.shape
#         _, in_channels, H_padded, W_padded = self.X.shape

#         # Initialize gradients
#         self.dW.fill(0)
#         self.db.fill(0)
#         dX_padded = np.zeros_like(self.X)

#         # Compute gradients
#         for i in range(new_H):
#             for j in range(new_W):
#                 h_start, h_end = i * self.stride, i * self.stride + self.kernel_size
#                 w_start, w_end = j * self.stride, j * self.stride + self.kernel_size

#                 # Extract input slice
#                 receptive_field = self.X[:, :, h_start:h_end, w_start:w_end]

#                 # Compute gradient w.r.t. weights
#                 self.dW += np.einsum('bchw,bo->ochw', receptive_field, d_out[:, :, i, j])

#                 # Compute gradient w.r.t. input
#                 dX_padded[:, :, h_start:h_end, w_start:w_end] += np.einsum('bo,ochw->bchw', d_out[:, :, i, j], self.W)

#         # Compute bias gradient
#         self.db = np.sum(d_out, axis=(0, 2, 3), keepdims=True)

#         # Remove padding from dX
#         if self.padding > 0:
#             self.dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
#         else:
#             self.dX = dX_padded

#         # Store gradients in 'grads' dictionary for optimizer
#         self.grads['W'] = self.dW
#         self.grads['b'] = self.db

#         self.dX.astype(np.float32)

#         return self.dX
    
#     def __call__(self, X):
#         X.astype(np.float32)
#         return self.forward(X)



# class conv2D(Layer):
#     """
#     The 2D convolutional layer.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
#                  initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.weight_decay = weight_decay
#         self.weight_decay_lambda = weight_decay_lambda

#         # Initialize weights and biases
#         self.W = initialize_method(size=(kernel_size, kernel_size))
#         self.b = initialize_method(size=(1, 1))

#         self.params = {'W': self.W, 'b': self.b}

#         # Gradients for W and b
#         self.grads = {'W': None, 'b': None}

#         # Store input for backward pass
#         self.input = None

#         self.optimizable=True

#     def __call__(self, X) -> np.ndarray:
#         return self.forward(X)

#     def forward(self, X):
#         """
#         Forward pass of the convolutional layer.
#         Input shape: [batch, in_channels, H, W]
#         Output shape: [batch, out_channels, new_H, new_W]
#         """

#         # Store input for backward pass
#         self.input = X

#         H, W = X.shape
#         kernel_size, _ = self.W.shape

#         # Compute output dimensions
#         new_H = (H + 2 * self.padding - kernel_size) // self.stride + 1
#         new_W = (W + 2 * self.padding - kernel_size) // self.stride + 1

#         # Apply padding if necessary
#         if self.padding > 0:
#             X_padded = np.pad(X, ((self.padding, self.padding), (self.padding, self.padding)),
#                               mode='constant', constant_values=0)
#         else:
#             X_padded = X

#         # Initialize output tensor
#         output = np.zeros((new_H, new_W))

#         # Perform convolution
#         for i in range(new_H):
#             for j in range(new_W):
#                 h_start = i * self.stride
#                 h_end = h_start + kernel_size
#                 w_start = j * self.stride
#                 w_end = w_start + kernel_size

#                 # Extract receptive field from input
#                 receptive_field = X_padded[h_start:h_end, w_start:w_end]

#                 # Convolve with weights and add bias
#                 output[:, :, i, j] = np.einsum('bchw,oikl->bo', receptive_field, self.W) + self.b.reshape(-1)

#         return output

#     def backward(self, grads):
#         """
#         Backward pass of the convolutional layer.
#         grads shape: [batch_size, out_channels, new_H, new_W]
#         """
#         batch_size, out_channels, new_H, new_W = grads.shape
#         _, in_channels, H, W = self.input.shape
#         _, _, kernel_size, _ = self.W.shape

#         # Initialize gradients for W and b
#         dW = np.zeros_like(self.W)
#         db = np.zeros_like(self.b)
#         dX = np.zeros_like(self.input)

#         # Pad input if necessary
#         if self.padding > 0:
#             padded_input = np.pad(self.input, ((self.padding, self.padding),
#                                                (self.padding, self.padding)), mode='constant', constant_values=0)
#         else:
#             padded_input = self.input

#         # Pad gradient for full convolution in backward pass
#         padded_grads = np.pad(grads, ((kernel_size - 1, kernel_size - 1),
#                                       (kernel_size - 1, kernel_size - 1)), mode='constant', constant_values=0)

#         # Compute gradients
#         for i in range(new_H):
#             for j in range(new_W):
#                 h_start = i * self.stride
#                 h_end = h_start + kernel_size
#                 w_start = j * self.stride
#                 w_end = w_start + kernel_size

#                 # Gradient for W
#                 receptive_field = padded_input[h_start:h_end, w_start:w_end]
#                 dW += np.einsum('bo,bchw->oikl', grads[i, j], receptive_field)

#                 # Gradient for X
#                 dX[h_start:h_end, w_start:w_end] += np.einsum('bo,oikl->bchw', grads[i, j], self.W)

#         # Gradient for b
#         db = np.sum(grads, axis=(0, 1), keepdims=True)

#         # Add L2 regularization if weight decay is enabled
#         if self.weight_decay:
#             dW += self.weight_decay_lambda * self.W

#         # Store gradients
#         self.grads['W'] = dW
#         self.grads['b'] = db

#         return dX

#     def clear_grad(self):
#         """
#         Clear gradients stored in self.grads.
#         """
#         self.grads = {'W': None, 'b': None}

class Squeeze(Layer):
    def __init__(self, axis=0):
        self.axis = axis
        self.optimizable =False
        self.input_shape=None

    def forward(self, X):
        """
        Forward pass: Remove single-dimensional entries from the shape of X.
        
        Args:
            X (numpy.ndarray): Input tensor.
            
        Returns:
            numpy.ndarray: Output tensor with squeezed dimensions.
        """
        self.input_shape = X.shape  # Save the input shape for backward pass
        return np.squeeze(X, axis=self.axis)
    
    def backward(self, grad_output):
        """
        Backward pass: Restore the squeezed dimensions in the gradient.
        
        Args:
            grad_output (numpy.ndarray): Gradient of the loss with respect to the output.
            
        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input.
        """
        # Restore the original shape by adding back the squeezed dimensions
        return np.reshape(grad_output, self.input_shape)
    
    def __call__(self, X):
        return self.forward(X)

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        X.astype(np.float32)
        return self.forward(X)

    def forward(self, X):
        X.astype(np.float32)
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

# 实现2D批量归一化层
class BatchNorm2D(Layer):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.params = {
            "gamma": np.ones((1, num_features, 1, 1)),
            "beta": np.zeros((1, num_features, 1, 1)),
        }
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        self.optimizable = True
        self.weight_decay=False

    def forward(self, X, training=True):
        if training:
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = X.var(axis=(0, 2, 3), keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            self.X_norm = (X - mean) / np.sqrt(var + self.epsilon)
        else:
            self.X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        return self.params["gamma"] * self.X_norm + self.params["beta"]

    def backward(self, grads):
        N, C, H, W = grads.shape
        gamma = self.params["gamma"]

        dX_norm = grads * gamma
        dvar = np.sum(dX_norm * (self.X_norm * -0.5) * np.power(self.running_var + self.epsilon, -1.5), axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dX_norm * -1 / np.sqrt(self.running_var + self.epsilon), axis=(0, 2, 3), keepdims=True) + dvar * np.sum(-2 * self.X_norm, axis=(0, 2, 3), keepdims=True) / (N * H * W)

        dX = dX_norm / np.sqrt(self.running_var + self.epsilon) + dvar * 2 * self.X_norm / (N * H * W) + dmean / (N * H * W)
        dgamma = np.sum(grads * self.X_norm, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        self.grads = {"gamma": dgamma, "beta": dbeta}
        return dX

    def __call__(self, X, training=True):
        return self.forward(X, training)

# 实现Dropout层
class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.optimizable = False

    def forward(self, X, training=True):
        if training:
            self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
            return X * self.mask
        return X

    def backward(self, grads):
        return grads * self.mask

    def __call__(self, X, training=True):
        return self.forward(X, training)


# Flatten the output of the last convolutional layer
class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.params = {}

        self.optimizable =False
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)
    
    def __call__(self, X):
        return self.forward(X)

class Reshape(Layer):
    def __init__(self, target_shape):
        """
        Initialize the Reshape layer.

        Args:
            target_shape (tuple): The target shape to reshape the input tensor to.
                                    Note: Does not include the batch dimension.
        """
        super().__init__()
        self.optimizable = False  # No trainable parameters in this layer
        self.target_shape = target_shape
        self.input_shape = None  # Store the input shape for backward pass

    def forward(self, X):
        """
        Forward pass: Reshape the input tensor to the target shape.

        Args:
            X (numpy.ndarray): Input tensor with shape (batch_size, ...).

        Returns:
            numpy.ndarray: Output tensor reshaped to (batch_size, *target_shape).
        """
        self.input_shape = X.shape  # Save the input shape for backward pass

        return X.reshape((X.shape[0], *self.target_shape))

    def backward(self, grad_output):
        """
        Backward pass: Restore the gradient to the original input shape.

        Args:
            grad_output (numpy.ndarray): Gradient of the loss with respect to the output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input.
        """
        # Restore the gradient to the original input shape
        return grad_output.reshape(self.input_shape)

    def __call__(self, X) -> np.ndarray:
        """
        Make the layer callable.

        Args:
            X (numpy.ndarray): Input tensor.

        Returns:
            numpy.ndarray: Output tensor after reshaping.
        """
        return self.forward(X)

class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        """
        初始化最大池化层
        :param kernel_size: 池化核的大小，默认为 2
        :param stride: 池化步幅，默认为 2
        :param padding: 输入的填充大小，默认为 0
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.optimizable = False
        self.input = None
        self.max_indices = None

    def forward(self, X):
        """
        计算最大池化层的前向传播
        :param X: 输入数据，形状为 (batch_size, in_channels, H, W)
        :return: 池化后的输出，形状为 (batch_size, in_channels, new_H, new_W)
        """
        self.input = X
        batch_size, in_channels, H, W = X.shape

        # 计算输出的 H 和 W
        new_H = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        new_W = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 如果有 padding，进行填充
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                            mode='constant', constant_values=0)
        else:
            X_padded = X

        # 输出池化结果
        output = np.zeros((batch_size, in_channels, new_H, new_W))
        # 保存最大值的索引，方便反向传播
        self.max_indices = np.zeros((batch_size, in_channels, new_H, new_W, self.kernel_size, self.kernel_size), dtype=bool)

        # 进行最大池化
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(new_H):
                    for j in range(new_W):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # 提取 receptive field
                        receptive_field = X_padded[b, c, h_start:h_end, w_start:w_end]

                        # 计算最大值
                        max_val = np.max(receptive_field)
                        output[b, c, i, j] = max_val

                        # 保存最大值对应的索引
                        self.max_indices[b, c, i, j, :, :] = (receptive_field == max_val)

        return output

    def backward(self, d_out):
        """
        计算最大池化层的反向传播
        :param d_out: 来自下一层的梯度，形状为 (batch_size, in_channels, new_H, new_W)
        :return: 当前层的输入的梯度，形状为 (batch_size, in_channels, H, W)
        """
        batch_size, in_channels, new_H, new_W = d_out.shape
        _, _, H, W = self.input.shape

        # 初始化梯度
        d_input = np.zeros_like(self.input)

        # 如果有 padding，计算原始输入在 padding 后的形状
        if self.padding > 0:
            H_padded = H + 2 * self.padding
            W_padded = W + 2 * self.padding
            d_input_padded = np.zeros((batch_size, in_channels, H_padded, W_padded))
        else:
            d_input_padded = d_input

        # 进行反向传播
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(new_H):
                    for j in range(new_W):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # 获取当前输出位置对应的梯度值
                        dout_val = d_out[b, c, i, j]

                        # 获取前向传播时保存的最大值索引
                        indices = self.max_indices[b, c, i, j, :, :]

                        # 将梯度传播到前向传播时最大值的位置
                        d_input_padded[b, c, h_start:h_end, w_start:w_end] += indices * dout_val

        # 移除 padding
        if self.padding > 0:
            d_input = d_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = d_input_padded

        return d_input

    def __call__(self, X):
        return self.forward(X)

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax.
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model  # Reference to the model for backward propagation
        self.max_classes = max_classes
        self.has_softmax = True  # Whether to apply softmax
        self.predicts = None  # Store predictions for backward pass
        self.labels = None  # Store labels for backward pass
        self.grads = None  # Gradients to be passed to the previous layer

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        Forward pass of the Cross-Entropy Loss.
        predicts: [batch_size, D] (logits before softmax)
        labels : [batch_size, ] (ground truth class indices)
        """
        self.predicts = predicts  # Save predictions for backward pass
        self.labels = labels      # Save labels for backward pass

        if self.has_softmax:
            # Apply softmax to convert logits into probabilities
            exp_predicts = np.exp(predicts - np.max(predicts, axis=1, keepdims=True))  # Numerical stability
            softmax_output = exp_predicts / np.sum(exp_predicts, axis=1, keepdims=True)
        else:
            softmax_output = predicts  # Assume input is already softmax probabilities

        # Compute cross-entropy loss
        batch_size = predicts.shape[0]
        log_probs = -np.log(np.maximum(softmax_output[np.arange(batch_size), labels], 1e-15))  # Avoid log(0)
        loss = np.mean(log_probs)  # Average loss over the batch

        return loss

    def backward(self):
        """
        Backward pass of the Cross-Entropy Loss.
        Computes gradients and sends them to the model for backpropagation.
        """
        batch_size = self.predicts.shape[0]

        if self.has_softmax:
            # Gradient of softmax + cross-entropy loss
            softmax_output = np.exp(self.predicts - np.max(self.predicts, axis=1, keepdims=True))
            softmax_output /= np.sum(softmax_output, axis=1, keepdims=True)

            # Initialize gradient
            self.grads = softmax_output.copy()

            # Subtract 1 from the correct class probabilities
            self.grads[np.arange(batch_size), self.labels] -= 1

            # Normalize gradient by batch size
            self.grads /= batch_size
        else:
            # Gradient of cross-entropy loss (without softmax)
            self.grads = np.zeros_like(self.predicts)
            self.grads[np.arange(batch_size), self.labels] = -1 / np.maximum(self.predicts[np.arange(batch_size), self.labels], 1e-15)
            self.grads /= batch_size

        # Pass gradients to the model for further backpropagation
        if self.model is not None:
            self.model.backward(self.grads)

    def cancel_softmax(self):
        """
        Cancel the internal softmax operation.
        Useful when the input to this layer is already softmax probabilities.
        """
        self.has_softmax = False
        return self
    
class ResidualBlock(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=True):
        super().__init__()
        self.conv1 = conv2D(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = ReLU()
        self.conv2 = conv2D(out_channels, out_channels, kernel_size, stride, padding)
        self.use_batchnorm = use_batchnorm
        self.optimizable=True
        self.params={'W1':self.conv1.W,'b1':self.conv1.b,'W2':self.conv2.W,'b2':self.conv2.b}
        self.X=None

        if use_batchnorm:
            self.bn1 = BatchNorm()
            self.bn2 = BatchNorm()
        
        # Shortcut Connection
        if in_channels != out_channels:
            self.shortcut = conv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = None

    def forward(self, X):
        identity = X  # 残差连接的输入
        self.X=X

        out = self.conv1(X)
        if self.use_batchnorm:
            out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        if self.use_batchnorm:
            out = self.bn2(out)
        
        # 添加 shortcut
        if self.shortcut:
            identity = self.shortcut(identity)

        out += identity  # 残差连接
        return self.relu(out)
    
    def backward(self, d_out):
        """
        反向传播，计算梯度。
        :param d_out: 上一层传来的梯度
        """
        d_out = self.relu.backward(d_out)  # 先通过 ReLU 反向传播

        identity = self.X  # 残差输入
        d_identity = d_out  # 直接传递梯度给 shortcut
        
        # 计算第二个卷积层的梯度
        d_out = self.conv2.backward(d_out)
        if self.use_batchnorm:
            d_out = self.bn2.backward(d_out)
        
        # 计算第一个卷积层的梯度
        d_out = self.relu.backward(d_out)
        d_out = self.conv1.backward(d_out)
        if self.use_batchnorm:
            d_out = self.bn1.backward(d_out)

        # 计算 shortcut 的梯度
        if self.shortcut:
            d_identity = self.shortcut.backward(d_identity)

        return d_out + d_identity  # 梯度相加，确保正确传播
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, epsilon=1e-5):
        """
        批归一化层 (支持 CNN 输入)
        :param momentum: 计算移动平均的动量参数
        :param epsilon: 避免除零错误的微小数值
        """
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon

        # 训练时统计的均值和方差
        self.running_mean = None
        self.running_var = None

        # 需要学习的参数
        self.gamma = None
        self.beta = None

        # 反向传播需要的中间变量
        self.cache = None
        self.optimizable = True
        self.params = {'gamma': self.gamma, 'beta': self.beta}
        self.grads = {'gamma': None, 'beta': None}

    def forward(self, X, training=True):
        """
        批归一化的前向传播 (适用于 CNN 格式)
        :param X: 输入数据，形状为 (batch_size, channels, height, width)
        :param training: 是否处于训练模式
        :return: 归一化后的输出
        """
        N, C, H, W = X.shape  # 解析输入形状

        # 初始化参数
        if self.gamma is None:
            self.gamma = np.ones((1, C, 1, 1))  # 形状 (1, C, 1, 1)
            self.beta = np.zeros((1, C, 1, 1))  # 形状 (1, C, 1, 1)
            self.params['gamma'] = self.gamma
            self.params['beta'] = self.beta
            self.grads['gamma'] = np.zeros_like(self.gamma)
            self.grads['beta'] = np.zeros_like(self.beta)
        if self.running_mean is None:
            self.running_mean = np.zeros((1, C, 1, 1))
            self.running_var = np.ones((1, C, 1, 1))

        if training:
            # 计算当前批次的均值和方差 (针对 C 维度计算)
            batch_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)  # 形状 (1, C, 1, 1)
            batch_var = np.var(X, axis=(0, 2, 3), keepdims=True)  # 形状 (1, C, 1, 1)

            # 归一化
            X_hat = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            out = self.gamma * X_hat + self.beta

            # 更新全局均值和方差（用于推理模式）
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            # 存储变量用于反向传播
            self.cache = (X, X_hat, batch_mean, batch_var, std_inv)
        else:
            # 直接使用移动平均值进行归一化
            X_hat = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * X_hat + self.beta

        return out

    def backward(self, d_out):
        """
        批归一化的反向传播 (适用于 CNN 格式)
        :param d_out: 上一层传来的梯度 (batch_size, channels, height, width)
        :return: 传回上一层的梯度
        """
        X, X_hat, mean, var, std_inv = self.cache
        N, C, H, W = X.shape  # 获取输入数据形状

        # 计算 d_beta
        d_beta = np.sum(d_out, axis=(0, 2, 3), keepdims=True)  # 形状 (1, C, 1, 1)
        self.grads['beta'] = d_beta

        # 计算 d_gamma
        d_gamma = np.sum(d_out * X_hat, axis=(0, 2, 3), keepdims=True)  # 形状 (1, C, 1, 1)
        self.grads['gamma'] = d_gamma

        # 计算 d_X
        d_X_hat = d_out * self.gamma  # 梯度传播到 X_hat
        d_var = np.sum(d_X_hat * (X - mean) * -0.5 * (var + self.epsilon)**(-1.5), axis=(0, 2, 3), keepdims=True)
        d_mean = np.sum(d_X_hat * (-std_inv), axis=(0, 2, 3), keepdims=True) + \
                 d_var * np.mean(-2.0 * (X - mean), axis=(0, 2, 3), keepdims=True)

        d_X = (d_X_hat * std_inv) + (d_var * 2 * (X - mean) / (N * H * W)) + (d_mean / (N * H * W))

        return d_X

    def __call__(self, X, training=True):
        return self.forward(X, training)


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