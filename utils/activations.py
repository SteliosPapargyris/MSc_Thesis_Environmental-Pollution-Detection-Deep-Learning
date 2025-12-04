import math

import torch as T


class PhotonicSinusoidal(T.nn.Module):

    def __init__(self, x_lower=0, x_upper=1):
        super().__init__()
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.y_upper = 2

    def forward(self, x):
        x = x.clamp(self.x_lower, self.x_upper)
        return T.pow(T.sin(x * math.pi / 2.0), self.y_upper)

    def __repr__(self):
        return 'PhotonicSinusoidal'


class PhotonicSigmoid(T.nn.Module):

    def __init__(self, A1=0.060, A2=1.005, x0=0.145, d=0.033, cutoff=2):
        super().__init__()
        self.A1 = A1
        self.A2 = A2
        self.x0 = x0
        self.d = d
        self.cutoff = cutoff

    def forward(self, x):
        x = x - self.x0
        x.clamp_(max=self.cutoff)
        return self.A2 + (self.A1 - self.A2) / (1 + T.exp(x / self.d))

    def __repr__(self):
        return 'PhotonicSigmoid'


class AdjustedReLUN(T.nn.Module):

    def __init__(self, n=2.0) -> None:
        super().__init__()

    def forward(self, x):
        return T.clamp_max(T.relu(x), self.bound)

    def __repr__(self):
        return "AdjustedReLU-{}".format(self.bound)


class ReLUN(T.nn.Module):

    def __init__(self, n=2.0):
        super().__init__()
        self.bound = n

    def forward(self, x):
        return T.clamp_max(T.relu(x), self.bound)

    def __repr__(self):
        return "ReLU-{}".format(self.bound)


class GELUN(T.nn.Module):
    __constants__ = ['approximate']
    approximate: str

    def __init__(self, approximate: str = 'none', n=6) -> None:
        super().__init__()
        self.approximate = approximate
        self.bound = n

    def forward(self, input: T.Tensor) -> T.Tensor:
        return T.nn.functional.gelu(T.clamp_max(input, self.bound),
                                    approximate=self.approximate)

    def extra_repr(self) -> str:
        return f'approximate={repr(self.approximate)}'
