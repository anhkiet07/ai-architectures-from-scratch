import numpy as np
class MSELoss:

    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        return ((self.pred - self.target) ** 2).mean()

    def backward(self):
        return 2 * (self.pred - self.target) / self.pred.size
    
