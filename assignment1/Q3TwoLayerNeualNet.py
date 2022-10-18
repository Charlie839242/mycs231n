import numpy as np
import matplotlib.pyplot as plt
from loss import *
from data_utils import load_CIFAR10


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network's architecture:
    input - fully connected layer - ReLU - fully connected layer - softmax_loss

    The input shape is N × D.
    The weight W1 shape is D × H. The bias b1 shape is H × 1.
    The activation is ReLU.
    The weight W2 shape is H × C. The bias b2 shape is C × 1.

    The equivalent equation: f(x) = [ Relu (X × W1 + b1) ] × W2 + b2

    The loss function was the composition of softmax loss and L2 regularization.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        params is to store initialized model parameters: W1, b1, W2, b2.
        graph includes the nodes in the computational graph.

        The input shape is N × D.
        The weight W1 shape is D × H. The bias b1 shape is H × 1.
        The activation is ReLU.
        The weight W2 shape is H × C. The bias b2 shape is C × 1.

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = dict()
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.dW1, self.dW2, self.db1, self.db2, self.dX = 0, 0, 0, 0, 0

        self.graph = dict()
        self.graph['fir_FC'] = MultiplyGate(self.params['W1'], self.params['b1'])
        self.graph['ReLU'] = ReLULayer()
        self.graph['sec_FC'] = MultiplyGate(self.params['W2'], self.params['b2'])
        self.graph['softmax'] = SoftmaxLayer()

    def forward(self, X, reg=None, y=None):
        """
        Do the forward pass
        Store the middle values for preparation of BP in self.graph

        The input shape is N × D.
        The weight W1 shape is D × H. The bias b1 shape is H × 1.
        The activation is ReLU.
        The weight W2 shape is H × C. The bias b2 shape is C × 1.
        """
        a = self.graph['fir_FC'].forward(X)
        b = self.graph['ReLU'].forward(a)
        output = self.graph['sec_FC'].forward(b)
        if type(y) != np.ndarray:
            return output
        else:
            loss = self.graph['softmax'].forward(output, self.graph['fir_FC'].W, self.graph['sec_FC'].W, reg, y)
            return output, loss

    def predict(self, X):
        scores = self.forward(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def validate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / y.shape[0]
        return accuracy

    def backward(self, lr):
        self.dW1, self.dW2, self.db1, self.db2, self.dX = 0, 0, 0, 0, 0
        tmp = self.graph['softmax'].backward(1)
        self.dX, self.dW1, self.dW2 = tmp[0], self.dW1 + tmp[1], self.dW2 + tmp[2]
        tmp = self.graph['sec_FC'].backward(self.dX)
        self.dX, self.dW2, self.db2 = tmp[0], self.dW2 + tmp[1], self.db2 + tmp[2]
        self.dX = self.graph['ReLU'].backward(self.dX)
        tmp = self.graph['fir_FC'].backward(self.dX)
        self.dX, self.dW1, self.db1 = tmp[0], self.dW1 + tmp[1], self.db1 + tmp[2]

        self.graph['fir_FC'].update(self.dW1, self.db1, lr)
        self.graph['sec_FC'].update(self.dW2, self.db2, lr)

    def train(self, X_train, y_train, X_test, y_test,
              lr=1e-4, reg=1e-5, num_iters=5000,
              batch_size=200, verbose=True, decay_rate=0.98):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X_train.shape[0]    # totally 50000 training images
        # use 200 images to train using SGD each time: 50000 / 200 = 250
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        for it in range(num_iters):
            shuffle_indexes = np.arange(num_train)
            np.random.shuffle(shuffle_indexes)
            shuffle_indexes = shuffle_indexes[0:batch_size - 1]
            X_batch = X_train[shuffle_indexes, :]
            y_batch = y_train[shuffle_indexes]

            _, loss = self.forward(X_batch, reg=reg, y=y_batch)
            self.backward(lr)
            loss_history.append(loss)

            if verbose and it % 50 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = self.validate(X_train, y_train)
                val_acc = self.validate(X_test, y_test)
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                print('Epoch %d : train_acc=%f, val_acc=%f' % (it / iterations_per_epoch, train_acc, val_acc))

                lr = lr * decay_rate

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }


if __name__ == "__main__":
    # numpy seed
    np.random.seed(1)

    # load cifar10 dataset
    cifar10_dir = 'cifar-10-batches-py'
    X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))
    X_test  = X_test .reshape((X_test.shape[0], X_test .shape[1] * X_test.shape[2]  * X_test.shape[3]))
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', Y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', Y_test.shape)

    # train
    net = TwoLayerNet(3072, 50, 10)
    detail = net.train(X_train, Y_train, X_test, Y_test, num_iters=20000)

    # visualize
    plt.plot(range(len(detail['loss_history'])), detail['loss_history'], label='loss')
    plt.show()
    plt.plot(range(len(detail['train_acc_history'])), detail['train_acc_history'], label='train_acc')
    plt.show()
    plt.plot(range(len(detail['val_acc_history'])), detail['val_acc_history'], label='val_acc_history')
    plt.show()



