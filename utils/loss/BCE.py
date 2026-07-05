import numpy as np
class BCELoss: 

    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        eps = 1e-8
        return -np.mean(self.target * np.log(self.pred + eps) + (1 - self.target) * np.log(1 - self.pred + eps))

    def backward(self):
        eps = 1e-8
        return (self.pred - self.target) / ((self.pred + eps) * (1 - self.pred + eps) * len(self.target))
