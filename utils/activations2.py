import math

import torch as T

### Photonic Sigmoid Constants ##
A1 = 0.060
A2 = 1.005
X_0 = 0.145
D = 0.033
CUTOFF = 2

### Photonic Siinusoidal Constants ##
X_LOWER = 0
X_UPPER = 1
Y_UPPER = 2


@staticmethod
def sigmoid_forward(x):
    x = x - X_0
    x.clamp_(max=CUTOFF)
    output = A2 + (A1 - A2) / (1.0 + T.exp(x / D))
    return output


class SigmoidRelu(T.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return sigmoid_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0

        return grad_input


class SigmoidSigmoid(T.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return sigmoid_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0

        return grad_input


class SigmoidRelu_1(T.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return sigmoid_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 1] = 0

        return grad_input


class SinusoidalRelu(T.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = x.clamp(X_LOWER, X_UPPER)
        output = T.pow(T.sin(x * math.pi / 2.0), Y_UPPER)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0

        return grad_input


class SinusoidalRelu_1(T.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = x.clamp(X_LOWER, X_UPPER)
        output = T.pow(T.sin(x * math.pi / 2.0), Y_UPPER)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 1] = 0

        return grad_input


class TanhLikeAF(T.nn.Module):

    def __init__(self, settings, gain=None):
        super().__init__()
        self.gain = gain
        self.sets = settings

    @classmethod
    def from_obs(cls, gain=3):
        EXP_OBS = dict([
            (3, {
                'a': 0.24057,
                'b': 0.34184,
                'c': 1.74544,
                'x0': -0.30873,
                'd': -1.65912,
                'e': 3.2698
            }),
            (5, {
                'a': 0.54474,
                'b': -0.00068,
                'c': 0.36802,
                'x0': -2.47491,
                'd': -6.47518,
                'e': 7.67979
            }),
            (7, {
                'a': 0.18333,
                'b': 0.62344,
                'c': 1.2335,
                'x0': 1.32589,
                'd': 0.66101,
                'e': 3.94583
            }),
        ])

        return cls(EXP_OBS[gain], gain)

    @classmethod
    def custom(cls,
               a=0.24057,
               b=0.34184,
               c=1.74544,
               x0=-0.30873,
               d=-1.65912,
               e=3.2698):
        return cls({'a': a, 'b': b, 'c': c, 'x0': x0, 'd': d, 'e': e})

    def forward(self, x):
        # x = x.clamp_min(-0.9)
        numer = self.sets['d'] + self.sets['b'] * T.sinh(x - self.sets['x0'])
        denum = self.sets['e'] + self.sets['c'] * T.cosh(x - self.sets['x0'])
        x = self.sets['a'] + numer / denum
        return x

    def __repr__(self):
        return 'Tanh-like AF - {}'.format(self.gain)


class EluLikeAF(T.nn.Module):

    def __init__(self, settings, gain=None):
        super().__init__()
        self.gain = gain
        self.sets = settings

    @classmethod
    def from_obs(cls, gain=3):
        EXP_OBS = dict([(3, {
            'a': 0.0368,
            'b': 0.18175,
            'c': -0.01957,
            'x0': 0.37042,
        }), (5, {
            'a': 0.10639,
            'b': 0.20202,
            'c': 0.00595,
            'x0': 0.35492
        }), (7, {
            'a': 0.21032,
            'b': 0.21012,
            'c': 0.06938,
            'x0': 0.55766
        })])

        return cls(EXP_OBS[gain], gain)

    @classmethod
    def custom(cls, a=0.0368, b=0.20202, c=0.06938, x0=0.35766):
        return cls({'a': a, 'b': b, 'c': c, 'x0': x0})

    def forward(self, x):

        x_lw_lmd = lambda x: self.sets['a'] * (T.exp(x - self.sets['x0']) - 1.0
                                               ) + self.sets['c']
        x_gr_lmd = lambda x: self.sets['b'] * (x - self.sets['x0']
                                               ) + self.sets['c']

        out = T.where(x < self.sets['x0'], x_lw_lmd(x), x_gr_lmd(x))

        return out

    def __repr__(self):
        return 'ELU-like AF - {}'.format(self.gain)


class InverseEluAF(T.nn.Module):

    def __init__(self, settings, gain=None):
        super().__init__()
        self.gain = None
        self.sets = settings

    @classmethod
    def from_obs(cls, gain=3):
        EXP_OBS = dict([
            (3, {
                'a': 0.02395,
                'b': 0.15568,
                'd': 0.04855,
                'e': 0.08616,
                'x0': -0.2
            }),
            (5, {
                'a': 0.09418,
                'b': 0.17363,
                'd': 0.04221,
                'e': 0.07855,
                'x0': 0.0
            }),
            (7, {
                'a': 0.15028,
                'b': 0.18086,
                'e': 0.06614,
                'd': 0.03925,
                'x0': 0.25
            }),
        ])

        return cls(EXP_OBS[gain], gain)

    @classmethod
    def custom(cls, a=0.02395, b=0.15568, e=0.08616, d=0.04855, x0=-0.2):
        return cls({'a': a, 'b': b, 'e': e, 'd': d, 'x0': x0})

    def forward(self, x):

        x_lw_lmd = lambda x: self.sets['b'] * (x - self.sets['x0']
                                               ) + self.sets['e']

        x_gr_lmd = lambda x: self.sets['a'] * (1. / (T.exp(-x + self.sets[
            'x0']) + 1.)) + self.sets['d']

        out = T.where(x > self.sets['x0'], x_gr_lmd(x), x_lw_lmd(x))

        return out

    def __repr__(self):
        return 'Inverse ELU-like AF - {}'.format(self.gain)


class ReSinAF(T.nn.Module):

    def __init__(self, settings, rf=None):
        super().__init__()
        self.rf = rf
        self.sets = settings

    @classmethod
    def from_obs(cls, rf=150):
        EXP_OBS = dict([
            (150, {
                'a': 0.23299,
                'b': 0.00047,
                'c': 0.01692,
                'x0': 0.44184,
                'k': -0.71482
            }),
            (400, {
                'a': 0.16258,
                'b': -0.00024,
                'c': 0.01326,
                'x0': 0.29698,
                'k': 0.91105
            }),
        ])

        return cls(EXP_OBS[rf], rf)

    @classmethod
    def custom(cls, a=0.16258, b=-0.00024, c=0.01326, x0=0.29698, k=0.91105):
        return cls({'a': a, 'b': b, 'c': c, 'x0': x0, 'k': k})

    def forward(self, x):
        x = x.clamp_max(2)
        x_lw_lmd = lambda x: self.sets['b'] * (x - self.sets['x0']
                                               ) + self.sets['c']
        x_gr_lmd = lambda x: self.sets['a'] * T.pow(
            T.sin(self.sets['k'] * (x - self.sets['x0'])), 2) + self.sets['c']

        x = T.where(x < self.sets['x0'], x_lw_lmd(x), x_gr_lmd(x))

        return x

    def __repr__(self):
        return 'ReSin AF - {}'.format(self.rf)


class ESinAF(T.nn.Module):

    def __init__(self, settings, rf=None):
        super().__init__()
        self.rf = rf
        self.sets = settings

    @classmethod
    def from_obs(cls, rf):
        EXP_OBS = dict([(150, {
            'a': 0.05125,
            'b': 0.00444,
            'c': 0.02348,
            'x0': -0.13453,
            'k': 0.57128
        })])

        return cls(EXP_OBS[rf], rf)

    @classmethod
    def custom(cls, a=0.05125, b=0.00444, c=0.02348, x0=-0.13453, k=0.57128):
        return cls({'a': a, 'b': b, 'c': c, 'x0': x0, 'k': k})

    def forward(self, x):
        x = x.clamp_max(2.6)
        x_lw_lmd = lambda x: self.sets['b'] * (T.exp(x) - 1.0) + self.sets['c']
        x_gr_lmd = lambda x: self.sets['a'] * T.pow(
            T.sin(self.sets['k'] * (x - self.sets['x0'])), 2) + self.sets['c']

        out = T.where(x < self.sets['x0'], x_lw_lmd(x), x_gr_lmd(x))

        return out

    def __repr__(self):
        return 'ESin AF - {}'.format(self.rf)


class SigmoidLikeAF(T.nn.Module):

    def __init__(self, settings, case=None, rf=None):
        super().__init__()
        self.case = case
        self.rf = rf
        self.sets = settings

    @classmethod
    def from_obs(cls, case, rf):
        EXP_OBS = {'left': {}, 'right': {}}

        EXP_OBS['left'] = dict([
            (150, {
                'a1': 0.0198,
                'a2': 0.07938,
                'x0': 1.26092,
                'dx': 0.48815
            }),
            (400, {
                'a1': 0.01188,
                'a2': 0.08372,
                'x0': 0.83367,
                'dx': 0.3052
            }),
            (600, {
                'a1': 0.00908,
                'a2': 0.0827,
                'x0': 0.69889,
                'dx': 0.23277
            }),
        ])

        EXP_OBS['right'] = dict([
            (150, {
                'a1': 0.01426,
                'a2': 0.06515,
                'x0': 0.48157,
                'dx': 0.56248
            }),
            (400, {
                'a1': 0.00709,
                'a2': 0.07334,
                'x0': 0.27768,
                'dx': 0.35234
            }),
            (600, {
                'a1': 0.00717,
                'a2': 0.07011,
                'x0': 0.18466,
                'dx': 0.2369
            }),
        ])

        return cls(EXP_OBS[case][rf], case, rf)

    @classmethod
    def custom(cls, a1=0.00717, a2=0.07011, x0=0.18466, dx=0.2369):
        return cls({'a1': a1, 'a2': a2, 'x0': x0, 'dx': dx})

    def forward(self, x):

        x = self.sets['a2'] + (self.sets['a1'] -
                               self.sets['a2']) / (1. + T.exp(
                                   (x - self.sets['x0']) / self.sets['dx']))

        return x

    def __repr__(self):
        return 'Sigmoid-like AF - {}/{}'.format(self.case, self.rf)


class DssAF(T.nn.Module):

    def __init__(self, settings, rf=None):
        super().__init__()
        self.sets = settings
        self.rf = rf

    @classmethod
    def from_obs(cls, rf):

        EXP_OBS = dict([(400, {
            'a': 0.10049,
            'b': 0.05079,
            'd': 0.85078,
            'c': 0.01014,
            'e': -0.84758,
            'x0': -0.02178
        })])

        return cls(EXP_OBS[rf], rf)

    @classmethod
    def custom(cls,
               a=0.10049,
               b=0.05079,
               d=0.85078,
               c=0.01014,
               e=-0.84758,
               x0=-0.02178):

        return cls({'a': a, 'b': b, 'd': d, 'c': c, 'e': e, 'x0': x0})

    def forward(self, x):
        x = x.clamp(-1.9, 1.9)
        x_lw_lmd = lambda x: self.sets['b'] * T.pow(
            T.sin(self.sets['e'] * (-x + self.sets['x0'])), 2) + self.sets['c']

        x_gr_lmd = lambda x: self.sets['a'] * T.pow(
            T.sin(self.sets['d'] * (x + self.sets['x0'])), 2) + self.sets['c']

        out = T.where(x < 0, x_lw_lmd(x), x_gr_lmd(x))
        return out

    def __repr__(self):
        return 'DSS AF - {}'.format(self.af)
