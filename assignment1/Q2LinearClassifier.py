import numpy as np
from data_utils import load_CIFAR10
from loss import svm_loss, softmax_loss


class LinearClassifier:
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-5, reg=1e-5, num_iters=2000,
                batch_size=200, verbose=True):
        """
        Inputs:
        - X: D x N array of training data. Each training point is a D-dimensional
             column. Eg: D=32*32*3=3072, N=50000
        - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        dim, num_train = X.shape        # X :3072×50000
        num_classes = np.max(y) + 1     # y takes values 0...K-1. Here K=10.

        # Initialize weight W: 10×3072
        if self.W is None:
            self.W = np.random.randn(num_classes, dim) * 0.001

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for i in range(num_iters):
            # randomly pick batch_size training data to do the training (the S in SGD)
            batch_indexes = np.random.choice(num_train, batch_size, replace=False)
            x_batch = X[:, batch_indexes]
            y_batch = y[batch_indexes]

            # Use randomly picked data to calculate the loss and gradient
            loss, grad = self.loss(x_batch, y_batch, reg)
            loss_history.append(loss)

            # Update weight W
            self.W = self.W - learning_rate * grad

            if verbose and i % 100 == 0:
                print('iteration %d / %d: loss %f' % (i, num_iters, loss))
        return loss_history

    def predict(self, x):
        y_pred = np.argmax(self.W.dot(x), 0)
        return y_pred

    def validate(self, x, y):
        pred = self.predict(x)
        accuracy = np.sum(pred==y) / y.shape[0]
        print('Validation Accuracy: ', accuracy)
        print(np.sum(pred==y))

    # loss function will be defined in sub-class. It should return loss and gradient.
    def loss(self, X_batch, y_batch, reg):
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """
    def loss(self, X_batch, y_batch, reg):
        return svm_loss(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss(self.W, X_batch, y_batch, reg)


if __name__=="__main__":
    # numpy seed
    np.random.seed(1)

    # load cifar10 dataset
    cifar10_dir = 'cifar-10-batches-py'
    X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)
    X_train = X_train.reshape((X_train.shape[1] * X_train.shape[2] * X_train.shape[3], X_train.shape[0]))
    X_test  = X_test .reshape((X_test .shape[1] * X_test.shape[2]  * X_test.shape[3] , X_test.shape[0]))
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', Y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', Y_test.shape)

    # svm = LinearSVM()
    # svm.train(X_train, Y_train)
    # svm.validate(X_train, Y_train)

    softmax = Softmax()
    softmax.train(X_train, Y_train)
    softmax.validate(X_train, Y_train)











