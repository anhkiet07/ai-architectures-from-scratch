class Tanh:
    def __call__(self, x):
        self.last_input = x
        return np.tanh(x)
    def backward(self, grad_output):
        return grad_output * (1 - self.last_input ** 2)
