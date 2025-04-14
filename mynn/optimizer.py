from abc import abstractmethod

import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    """
    Momentum Gradient Descent Optimizer based on the formula:
    v^{t+1} = beta * v^t + grad_f(w^t)
    w^{t+1} = w^t - alpha_t * v^{t+1}
    """
    def __init__(self, init_lr, model, beta=0.9):
        super().__init__(init_lr, model)
        self.lr = init_lr
        self.beta = beta  # Momentum factor (usually between 0 and 1)
        self.velocity = {}  # Store velocity for momentum term

        # Initialize velocity for each optimizable layer
        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                layer_name = getattr(layer, 'name', f"layer_{idx}")  # Use 'name' if exists, else use index
                self.velocity[layer_name] = {}
                for key in layer.params.keys():
                    self.velocity[layer_name][key] = np.zeros_like(layer.params[key])

    def update_lr(self, new_lr):
        """
        Update the learning rate of the optimizer.
        :param new_lr: New learning rate to be set.
        """
        self.lr = new_lr  # Update the learning rate

    def step(self):
        """
        Update the parameters using the given formula.
        """
        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                layer_name = getattr(layer, 'name', f"layer_{idx}")  # Use 'name' if exists, else use index
                for key in layer.params.keys():
                    # Ensure the velocity and gradients have matching shapes
                    velocity_shape = self.velocity[layer_name][key].shape
                    grad_shape = layer.grads[key].shape

                    if velocity_shape != grad_shape:
                        # If the shapes are different, reshape the gradients to match the velocity shape
                        layer.grads[key] = np.reshape(layer.grads[key], velocity_shape)

                    # Update velocity: v^{t+1} = beta * v^t + grad_f(w^t)
                    self.velocity[layer_name][key] = (
                        self.beta * self.velocity[layer_name][key] + layer.grads[key]
                    )

                    # Apply weight decay if enabled
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.lr * layer.weight_decay_lambda)

                    # Update parameters: w^{t+1} = w^t - alpha_t * v^{t+1}
                    layer.params[key] -= self.lr * self.velocity[layer_name][key]