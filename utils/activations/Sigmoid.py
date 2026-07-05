class Sigmoid:
    def __call__(self, x):
        self.last_input = x
        return 1 / (1 + np.exp(-x))
    def backward(self, grad_output):
        return grad_output * (self.last_input * (1 - self.last_input))
