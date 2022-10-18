import numpy as np
from data_utils import load_CIFAR10


class KNearestNeighbor:
    def __init__(self):
        self.X_train = None
        self.Y_train = None

    def train(self, x, y):  # 储存训练数据
        self.X_train = x
        self.Y_train = y

    def l1_distance(self, x):   # 计算l1距离
        distance = np.sum(np.abs(self.X_train - x), axis=(1,2,3))
        return distance

    def l2_distance(self, x):   # 计算l2距离
        distance = np.sqrt(np.sum(np.square(self.X_train - x), axis=(1,2,3)))
        return distance

    def find_nearest_k_points(self, distance, k):   # 找到距离中最小k个距离的索引
        index = np.argpartition(distance, k)[:k]
        return index

    def vote(self, index):  # 判断k个周围点中最多的种类
        neighbor = self.Y_train[index]
        return np.argmax(np.bincount(neighbor))  # 找到出现最多的元素(bincount返回从0到最大值出现的次数)

    def predict(self, x, k, distance_type):
        if distance_type == 'l1':
            distance = self.l1_distance(x)
        elif distance_type == 'l2':
            distance = self.l2_distance(x)
        else:
            raise ValueError('Distance_type should be l1 or l2.')
        index = self.find_nearest_k_points(distance, k)
        type = self.vote(index)
        return type

    def validate(self, x_test, y_test, k, distance_type):
        correction = 0
        print('Total number to validate is : ', x_test.shape[0])
        for m, n in zip(x_test, y_test):
            correction = correction + 1 if self.predict(m, k, distance_type) == n else correction
        correction_rate = correction / x_test.shape[0]
        return correction_rate


if __name__=="__main__":
    # numpy seed
    np.random.seed(1)

    # load cifar10 dataset
    cifar10_dir = 'cifar-10-batches-py'
    X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', Y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', Y_test.shape)

    KNN = KNearestNeighbor()
    KNN.train(X_train[0:5000], Y_train[0:5000]) # 为了计算方便只取前5000张
    print('k=1时，用l1距离的准确率：', KNN.validate(X_test[0:100], Y_test[0:100], 1, 'l1'))
    print('k=1时，用l2距离的准确率：', KNN.validate(X_test[0:100], Y_test[0:100], 1, 'l2'))
    print('k=7时，用l1距离的准确率：', KNN.validate(X_test[0:100], Y_test[0:100], 7, 'l1'))
    print('k=7时，用l2距离的准确率：', KNN.validate(X_test[0:100], Y_test[0:100], 7, 'l2'))







