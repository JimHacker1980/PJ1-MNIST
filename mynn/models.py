import pickle

from .op import *


class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    CNN Model using custom layers.
    """
    def __init__(self):
        super().__init__()
        self.layers = []  # Store model layers
        self.params = {}  # Trainable parameters

        # Convolutional Block 1
        self.add_layer(conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1))
        self.add_layer(ReLU())
        self.add_layer(BatchNorm2D(32))
        self.add_layer(conv2D(in_channels=32, out_channels=32, kernel_size=3, padding=1))
        self.add_layer(ReLU())
        self.add_layer(BatchNorm2D(32))
        self.add_layer(MaxPool2D(kernel_size=2, stride=2))

        # Convolutional Block 2
        self.add_layer(conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1))
        self.add_layer(ReLU())
        self.add_layer(BatchNorm2D(64))
        self.add_layer(conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1))
        self.add_layer(ReLU())
        self.add_layer(BatchNorm2D(64))
        self.add_layer(MaxPool2D(kernel_size=2, stride=2))

        # Fully Connected Layer
        self.add_layer(Flatten())
        self.add_layer(Linear(in_dim=64 * 7 * 7, out_dim=512))
        self.add_layer(ReLU())
        self.add_layer(Dropout(0.5))
        self.add_layer(Linear(in_dim=512, out_dim=10))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                loss_grad = layer.backward(loss_grad)

    def add_layer(self, layer):
        self.layers.append(layer)
        if hasattr(layer, 'params'):
            self.params.update(layer.params)

    def load_model(self, param_list):
        for layer_name, params in param_list.items():
            if layer_name in self.params:
                self.params[layer_name]['W'] = params['W']
                self.params[layer_name]['b'] = params['b']

    def save_model(self, save_path):
        np.save(save_path, self.params)

    def clear_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'clear_grad'):
                layer.clear_grad()
