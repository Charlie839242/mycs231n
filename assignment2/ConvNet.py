import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_CIFAR10
from layer_class import *


class ConvNet(object):
    """
    The structure of ConvNet is:
    Conv - Maxpool - Relu - Conv - Maxpool - Relu - Affine - Softmax
    """
    def __init__(self):
        self.graph = dict()
        self.graph['Layer_1'] = Conv(filter_size=(16, 3, 3, 3), stride=1, pad=1)
        self.graph['Layer_2'] = Maxpool(pool_width=2, pool_height=2, stride=2)
        self.graph['Layer_3'] = Relu()
        self.graph['Layer_4'] = Conv(filter_size=(32, 16, 3, 3), stride=1, pad=1)
        self.graph['Layer_5'] = Maxpool(pool_width=2, pool_height=2, stride=2)
        self.graph['Layer_6'] = Relu()
        self.graph['Layer_7'] = Affine(w_size=(2048, 10))
        self.graph['Layer_8'] = Softmax()

    def forward(self, x, y=None):
        # Assume input x has shape (N, 3, 32, 32)
        x = self.graph['Layer_1'].forward(x)        # (N, 16, 32, 32)
        x = self.graph['Layer_2'].forward(x)        # (N, 16, 16, 16)
        x = self.graph['Layer_3'].forward(x)        # (N, 16, 16, 16)
        x = self.graph['Layer_4'].forward(x)        # (N, 32, 16, 16)
        x = self.graph['Layer_5'].forward(x)        # (N, 32, 8 , 8 )
        x = self.graph['Layer_6'].forward(x)        # (N, 32, 8 , 8 )
        out = self.graph['Layer_7'].forward(x)      # (N, 10)

        mode = 'test' if type(y) != np.ndarray else 'train'
        if mode == 'test':
            return out

        loss = self.graph['Layer_8'].forward(out, y)
        return out, loss

    def predict(self, x):
        scores = self.forward(x)                    # (N, 10)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def validate(self, x, y):
        y_pred = self.predict(x)
        accuracy = np.sum(y_pred == y) / y.shape[0]
        return accuracy

    def backward(self):
        dx_8 = self.graph['Layer_8'].backward()
        dx_7 = self.graph['Layer_7'].backward(dx_8)
        dx_6 = self.graph['Layer_6'].backward(dx_7)
        dx_5 = self.graph['Layer_5'].backward(dx_6)
        dx_4 = self.graph['Layer_4'].backward(dx_5)
        dx_3 = self.graph['Layer_3'].backward(dx_4)
        dx_2 = self.graph['Layer_2'].backward(dx_3)
        dx_1 = self.graph['Layer_1'].backward(dx_2)

    def update(self, lr=1e-2):
        self.graph['Layer_1'].update(lr)
        self.graph['Layer_4'].update(lr)
        self.graph['Layer_7'].update(lr)

    def train(self, X_train, y_train, X_test, y_test, num_iters=5000,
              batch_size=200, verbose=True, decay_rate=0.98):
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

            _, loss = self.forward(X_batch, y_batch)
            self.backward()
            self.update()
            loss_history.append(loss)

            if verbose and it % 50 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = self.validate(X_train[0:1000, :], y_train[0:1000])
                val_acc = self.validate(X_test[0:1000, :], y_test[0:1000])
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                print('Epoch %d : train_acc=%f, val_acc=%f' % (it / iterations_per_epoch, train_acc, val_acc))

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
    X_train = X_train.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', Y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', Y_test.shape)

    # train
    net = ConvNet()
    detail = net.train(X_train, Y_train, X_test, Y_test, num_iters=20000)

    # visualize
    plt.plot(range(len(detail['loss_history'])), detail['loss_history'], label='loss')
    plt.show()
    plt.plot(range(len(detail['train_acc_history'])), detail['train_acc_history'], label='train_acc')
    plt.show()
    plt.plot(range(len(detail['val_acc_history'])), detail['val_acc_history'], label='val_acc_history')
    plt.show()