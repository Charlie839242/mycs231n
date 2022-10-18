from layers import *


class Conv(object):
    def __init__(self, filter_size, stride, pad, std=1e-4):
        self.K, self.C, self.F_H, self.F_W = filter_size
        self.w = std * np.random.randn(self.K, self.C, self.F_H, self.F_W)
        self.b = std * np.random.randn(self.K)
        conv_param = dict()
        conv_param['stride'], conv_param['pad'] = stride, pad
        self.conv_param = conv_param
        self.cache = None
        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        out, cache = conv_forward_naive(x, self.w, self.b, self.conv_param)
        self.cache = cache
        return out

    def backward(self, up_grad):
        dx, dw, db = conv_backward_naive(up_grad, self.cache)
        self.dx, self.dw, self.db = dx, dw, db
        return dx

    def update(self, lr):
        self.w = self.w - lr * self.dw
        self.b = self.b - lr * self.db


class Maxpool(object):
    def __init__(self, pool_width, pool_height, stride):
        pool_param = dict()
        pool_param['pool_width'], pool_param['pool_height'], pool_param['stride'] = pool_width, pool_height, stride
        self.pool_param = pool_param
        self.cache = None
        self.dx = None

    def forward(self, x):
        out, cache = max_pool_forward_naive(x, self.pool_param)
        self.cache = cache
        return out

    def backward(self, up_grad):
        dx = max_pool_backward_naive(up_grad, self.cache)
        self.dx = dx
        return dx


class Relu(object):
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        X[X < 0] = 0
        return X

    def backward(self, up_grad):
        dX = self.X
        dX[dX > 0] = 1
        dX[dX < 0] = 0
        return dX * up_grad


class Affine(object):
    def __init__(self, w_size, std=1e-4):
        self.Height, self.Width = w_size
        self.w = std * np.random.randn(self.Height, self.Width)
        self.b = std * np.random.randn(self.Width)
        self.cache = None
        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        out, cache = affine_forward(x, self.w, self.b)
        self.cache = cache
        return out

    def backward(self, up_grad):
        dx, dw, db = affine_backward(up_grad, self.cache)
        self.dx, self.dw, self.db = dx, dw, db
        return dx

    def update(self, lr):
        self.w = self.w - lr * self.dw
        self.b = self.b - lr * self.db


class Softmax(object):
    def __init__(self):
        self.dx = None

    def forward(self, x, y):
        loss, dx = softmax_loss(x, y)
        self.dx = dx
        return loss

    def backward(self):
        return self.dx




