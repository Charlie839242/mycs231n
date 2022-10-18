import numpy as np

"""
svm_loss and softmax_loss 是计算整个以svm和softmax为损失函数的线性分类器的损失和梯度的函数。
所以svm_loss and softmax_loss只只用于Q2里面。
对于Q3里面的梯度计算，可以按照节点来定义每个节点的local gradient并利用链式法则计算。
"""


def svm_loss(W, X, y, reg):
    """
    SVM loss and gradient:
    输入X: 3072 × 1, W是10 × 3072, W_j代表W的第j行.
    输出是10个数（范围不定，经过softmax后范围才是0到1），越大越有可能是这一类别。
    这10个数分别用s_j表示，j表示对应的类别, j ranges from 1 to 10
    ground truth label 是yi.
    loss = Σ_(j≠yi) max(0, s_j-s_yi+1)
    也就是在所有非标签的数s_j与s_yi作max(0, s_j-s_yi+1)并求和
    然后是gradient:
        j = yi: (d loss/d W_yi) = {Σ_(j≠yi) [1(s_j-s_yi+1>0)]} × (-x)
        j ≠ yi: (d loss/d W_j)  = [1(s_j-s_yi+1>0)] × (x)
    """
    num_train = X.shape[1]                      # X: 3072 × n, n是每次随机梯度下降时取了mini-batch的数量
    output = W.dot(X)                           # W: 10 × 3072, output: 10 × n
    sequence = np.array(range(num_train))
    correct_class_scores = output[y, sequence]  # correct_class_scores: 1 × n

    margins = np.maximum(0, output - correct_class_scores + 1)  # max(0, s_j-s_yi+1) 但也计算了标签那一行
                                                                # margins: 10 × n
    margins[y, sequence] = 0        # 把标签那一行置零
    loss = np.sum(margins)          # 把所有loss求和

    # calculate the loss
    loss = loss / num_train
    loss = loss + 0.5 * reg * np.sum(W * W)     # L2 regularization: W * W 即 Σ(w^2)

    # calculate the gradient
    # margins: 10 × n.  W: 10 × 3072
    # margins[i, m]负责更新W[i], margins[:, m]更新整个W. 用m遍历range(n)得到n个更新值，取平均即可.
    # 相当于对margins的每一列作处理得到n个结果，n个结果取平均.
    margins = np.where(margins > 0, 1, 0)                   # 对于每一列j≠yi的位置, 取1(s_j-s_yi+1>0)
    margins[y, sequence] = -1 * np.sum(margins, axis=0)     # 对于每一列j=yi的位置, 取 -Σ_(j≠yi) [1(s_j-s_yi+1>0)]
    dW = np.dot(margins, X.T)                               # dW: 10 × 3072     这个是n个输入得到的dW, 所以最后还要除以n

    dW = dW / num_train
    dW = dW + reg * W                                       # gradient of the L2 regularization is W itself

    return loss, dW


def softmax_loss(W, X, y, reg):
    """
    Softmax loss and gradient:
    输入X: 3072 × 1, W是10 × 3072, W_j代表W的第j行.
    输出是10个数（范围不定，经过softmax后范围才是0到1），越大越有可能是这一类别。
    这10个数分别用s_j表示，j表示对应的类别, j ranges from 1 to 10
    ground truth label 是yi.
    loss = -log[(e^s_yi)/(Σ_j e^s_j)]

    然后是gradient:
        j = yi: (d loss/d W_yi) = [(e^s_yi)/(Σ_j e^s_j)-1] × x
        j ≠ yi: (d loss/d W_j)  = [(e^s_j)/(Σ_j e^s_j)]    × x
    """
    num_train = X.shape[1]              # X: 3072 × n, n是每次随机梯度下降时取了mini-batch的数量

    output = W.dot(X)                   # W: 10 × 3072, output: 10 × n
    exp_scores = np.exp(output)         # 全部取指数
    prob_scores = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)  # normalize so that np.sum(prob_scores[:,x])=1
    loss = np.sum(-np.log(prob_scores[y, range(num_train)]))                  # 把n个标签位置再取对数相加即是total softmax loss
    loss = loss / num_train
    loss = loss + 0.5 * reg * np.sum(W ** 2)                                  # L2 regularization

    # gradient
    prob_scores[y, range(num_train)] = prob_scores[y, range(num_train)] - 1   # 根据公式，标签位置的gradient要再-1
    dW = np.dot(prob_scores, X.T)
    dW = dW / num_train
    dW = dW + reg * W

    return loss, dW


"""
因为之后想要自己实现一个像pytorch的框架，所以这里熟悉一下怎么写计算图。
对于Q4，尝试用模块化的计算图来做反向传播。
下面来实现一个全连接层, 一个ReLU激活层和一个softmax层。
整个Q4就是由: MultiplyGate - ReLULayer - MultiplyGate 来组成的
"""


class MultiplyGate(object):
    """
    example:
        W: D × H
        X: N × D
        output: N × H
    """
    def __init__(self, W, b):
        self.W = W
        self.X = None
        self.b = b

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W)+self.b

    def backward(self, up_grad):
        dX = np.dot(up_grad, self.W.T)
        dW = np.dot(self.X.T, up_grad)
        db = np.sum(up_grad, axis=0)
        return dX, dW, db

    def update(self, dW, db, lr):
        self.W = self.W - lr * dW
        self.b = self.b - lr * db


class ReLULayer(object):
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


class SoftmaxLayer(object):
    def __init__(self):
        self.X = None
        self.W1 = None
        self.W2 = None
        self.reg = None
        self.y = None

    def forward(self, X, W1, W2, reg, y):
        self.X = X
        self.W1 = W1
        self.W2 = W2
        self.reg = reg
        self.y = y
        softmax_loss = -np.log(np.exp(X[range(len(y)), y]) / np.sum(np.exp(X), axis=1))
        total_loss = np.sum(softmax_loss) / len(y) + reg * 0.5 * np.sum(W1 ** 2) + reg * 0.5 * np.sum(W2 ** 2)
        return total_loss

    def backward(self, up_grad):
        exp_scores = np.exp(self.X)  # 全部取指数
        prob_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # exp to probabilities
        prob_scores[range(len(self.y)), self.y] = prob_scores[range(len(self.y)), self.y] - 1  # 根据公式，标签位置的gradient要再-1
        dX = np.dot(prob_scores, up_grad) / len(self.y)
        dW1 = self.reg * self.W1
        dW2 = self.reg * self.W2
        return dX, dW1, dW2









