import numpy as np
class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = None
    def __init__(self, in_features, out_features, activation = None):
        self.w = Parameter(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features))
        self.W = Parameter(np.random.randn(in_features, out_features) * np.sqrt(1. / in_features))
        self.b = Parameter(np.zeros((1, out_features)))
    def forward(self, x):
        self.last_input = x
        return x @ self.w.data + self.b.data
    def backward(self, grad_output):
        self.w.grad = self.last_input.T @ grad_output
        self.b.grad = np.sum(grad_output, axis=0, keepdims=True)
        self.W.grad = grad_w if self.W.grad is None else self.W.grad + grad_w
        self.b.grad = grad_b if self.b.grad is None else self.b.grad + grad_b
        grad_input = grad_output @ self.w.data.T
        return grad_input
