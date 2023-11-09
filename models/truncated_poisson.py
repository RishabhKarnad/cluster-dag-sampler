import numpy as np
import scipy.stats as stats


class TruncatedPoisson:
    def __init__(self, min, mean, max):
        self.min = min
        self.mean = mean
        self.max = max
        self.P = stats.poisson(self.mean)

    def logpmf(self, x):
        return self.P.logpmf(x) - np.log(np.sum([self.P.pmf(xi) for xi in range(self.min, self.max+1)]))

    def pmf(self, x):
        return np.exp(self.logpmf(x))

    def sample(self):
        raise NotImplementedError
