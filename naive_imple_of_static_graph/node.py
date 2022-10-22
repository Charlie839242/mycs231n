from naive_graph import naive_Graph
import random


"""
Tensorflow中有三种变量：常量（Constant）；变量（Variable）；占位符（Placeholder）。
常量不存在导数，求导通常是对变量和占位符去求，而占位符通常是神经网络的数据输入口，
因此继承naive_Graph.Node来设计这三个变量的派生类。
每个派生类都包含了其前向传播的value和反向传播的grad（除了Constant）
"""


class Constant(naive_Graph.Node):
    def __init__(self, value):
        super().__init__()
        self.__value = float(value)

    def get_value(self):
        return self.__value

    def __repr__(self):
        return str(self.__value)


class Variable(naive_Graph.Node):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)
        self.grad = 0.

    def get_value(self):
        return self.value

    def __repr__(self):
        return str(self.value)


class PlaceHolder(naive_Graph.Node):
    def __init__(self):
        super().__init__()
        self.value = None
        self.grad = 0.

    def get_value(self):
        return self.value

    def __repr__(self):
        return str(self.value)


if __name__ == "__main__":
    random.seed(1)
    c = Constant(1)
    print(c.id)
    print(c)
