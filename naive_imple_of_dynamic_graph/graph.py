from math import prod, e
from math import log as math_log, pow as math_pow
from math import sin as math_sin, cos as math_cos
import matplotlib.pyplot as plt


class Graph:
    # node_list只保存计算图里的节点(也就是反向传播需要的节点)
    node_list = []

    forward_calculate_table = {
        # 算子的前向传播
        "add": lambda node1, node2: node1.value + node2.value,
        "sub": lambda node1, node2: node1.value - node2.value,
        "div": lambda node1, node2: node1.value / node2.value,
        "pow": lambda node1, node2: math_pow(node1.value, node2.value),
        "mul": lambda node1, node2: node1.value * node2.value,
        "log": lambda node1: math_log(node1.value),
        "sin": lambda node1: math_sin(node1.value),
        "cos": lambda node1: math_cos(node1.value)
    }

    @classmethod
    def add_node(cls, node):
        cls.node_list.append(node)

    @classmethod
    def clear(cls):
        cls.node_list.clear()

    @classmethod
    def free_graph(cls):
        """销毁计算图"""
        new_list = []
        for node in Graph.node_list:
            node.next.clear()
            node.last.clear()
            if node.is_leaf:
                # 叶子节点
                new_list.append(node)
        Graph.node_list = new_list

    @classmethod
    def kill_grad(cls):
        # 清空所有梯度
        for node in cls.node_list:
            node.grad = 0

    class Node(object):
        def __init__(self, value, requires_grad=False, grad_fn=None):
            self.value = float(value)
            self.requires_grad = requires_grad
            self.grad = 0. if self.requires_grad else None
            self.grad_fn = grad_fn if self.requires_grad else None
            self.is_leaf = True if self.grad_fn is None else False
            # 默认是操作符名，该属性为绘图需要
            self.name = None
            self.next = []
            self.last = []
            # 由于不需要前向传播，所以入度属性被淘汰
            self.out_deg, self.out_deg_com = 0, 0
            if self.requires_grad:
                # 不需要求梯度的节点不出现在动态计算图中
                Graph.add_node(self)

        def build_edge(self, node):
            # 构建self指向node的有向边
            self.out_deg += 1
            self.next.append(node)
            node.last.append(self)

        """ 重构Node运算符 """
        def __add__(self, node):    # 重构加法: self出现在加号左边
            return Graph.add(self, node)

        def __radd__(self, node):
            return Graph.add(node, self)

        def __mul__(self, node):    # 重构乘法
            return Graph.mul(self, node)

        def __rmul__(self, other):
            return Graph.mul(other, self)

        def __truediv__(self, node):    # 重构除法
            return Graph.div(self, node)

        def __sub__(self, node):    # 重构减法
            return Graph.sub(self, node)

        def __pow__(self, node):
            return Graph.pow(self, node)

        def __rpow__(self, node):  # 重构指数
            return Graph.pow(node, self)

        def sin(self):
            return Graph.sin(self)

        def cos(self):
            return Graph.cos(self)

        def log(self):
            return Graph.log(self)

        def backward(self, retain_graph=False):
            if self not in Graph.node_list:
                print("AD failed because the node is not in graph")
                return

            node_queue = []
            self.grad = 1.

            for node in Graph.node_list:
                if node.requires_grad:
                    if node.out_deg == 0:
                        node_queue.append(node)

            while len(node_queue) > 0:
                node = node_queue.pop()
                for last_node in node.last:
                    last_node.out_deg -= 1
                    last_node.out_deg_com += 1
                    if last_node.out_deg == 0 and last_node.requires_grad:
                        # 梯度向requires_grad=True的上游节点流去
                        for n in last_node.next:
                            last_node.grad += n.grad * n.grad_fn(n, last_node)
                        node_queue.insert(0, last_node)

            if retain_graph:
                # 保留图，即恢复所有node的出度
                for node in Graph.node_list:
                    node.out_deg += node.out_deg_com
                    node.out_deg_com = 0
            else:
                # 释放计算图：删除所有非叶子节点
                Graph.free_graph()

    """反向传播算子"""
    @classmethod
    def deriv_add(cls, child, parent):
        return 1

    @classmethod
    def deriv_sub(cls, child, parent):
        if child.last[0] == parent:
            return 1
        elif child.last[1] == parent:
            return -1

    @classmethod
    def deriv_mul(cls, child, parent):
        if child.last[0] == parent:
            return child.last[1].value
        elif child.last[1] == parent:
            return child.last[0].value

    @classmethod
    def deriv_div(cls, child, parent):
        if child.last[0] == parent:
            return 1 / child.last[1].value
        elif child.last[1] == parent:
            return -child.last[0].value / (child.last[1].value ** 2)

    @classmethod
    def deriv_pow(cls, child, parent):
        if child.last[0] == parent:
            return child.last[1].value * math_pow(child.last[0].value, child.last[1].value - 1)
        elif child.last[1] == parent:
            return child.value * math_log(child.last[0].value)

    @classmethod
    def deriv_log(cls, child, parent):
        return 1 / parent.value

    @classmethod
    def deriv_sin(cls, child, parent):
        return math_cos(parent.value)

    @classmethod
    def deriv_cos(cls, child, parent):
        return -math_sin(parent.value)

    backward_calculate_table = {
        "add": lambda child, parent: Graph.deriv_add(child, parent),
        "sub": lambda child, parent: Graph.deriv_sub(child, parent),
        "div": lambda child, parent: Graph.deriv_div(child, parent),
        "pow": lambda child, parent: Graph.deriv_pow(child, parent),
        "mul": lambda child, parent: Graph.deriv_mul(child, parent),
        "log": lambda child, parent: Graph.deriv_log(child, parent),
        "sin": lambda child, parent: Graph.deriv_sin(child, parent),
        "cos": lambda child, parent: Graph.deriv_cos(child, parent),
    }

    grad_fn_table = {
        # 将求值公式和求导公式存在字典grad_fn_table里面
        "add": (forward_calculate_table.get('add'), backward_calculate_table.get('add')),
        "mul": (forward_calculate_table.get('mul'), backward_calculate_table.get('mul')),
        "div": (forward_calculate_table.get('div'), backward_calculate_table.get('div')),
        "sub": (forward_calculate_table.get('sub'), backward_calculate_table.get('sub')),
        "pow": (forward_calculate_table.get('pow'), backward_calculate_table.get('pow')),
        "log": (forward_calculate_table.get('log'), backward_calculate_table.get('log')),
        "sin": (forward_calculate_table.get('sin'), backward_calculate_table.get('sin')),
        "cos": (forward_calculate_table.get('cos'), backward_calculate_table.get('cos')),
    }

    @classmethod
    def unary_function_frame(cls, node, operator: str):
        if type(node) != cls.Node:
            node = cls.Node(node)

        # grad_fn_table是一个字符串——函数元组字典，元组中是求值函数和求导函数
        fn, grad_fn = cls.grad_fn_table.get(operator)
        # 这里fn(node)说明我们直接计算输出，即动态计算图的特征
        operator_node = cls.Node(fn(node), requires_grad=node.requires_grad, grad_fn=grad_fn)
        operator_node.name = operator
        if operator_node.requires_grad:
            # 只有可求导的变量间才会用有向边联系
            node.build_edge(operator_node)
        return operator_node

    @classmethod
    def binary_function_frame(cls, node1, node2, operator):
        if type(node1) != cls.Node:
            node1 = cls.Node(node1)
        if type(node2) != cls.Node:
            node2 = cls.Node(node2)

        fn, grad_fn = cls.grad_fn_table.get(operator)
        requires_grad = node1.requires_grad or node2.requires_grad
        operator_node = cls.Node(fn(node1, node2), requires_grad=requires_grad, grad_fn=grad_fn)
        operator_node.name = operator
        if requires_grad:
            node1.build_edge(operator_node)
            node2.build_edge(operator_node)
        return operator_node

    @classmethod
    def add(cls, node1, node2):
        return Graph.binary_function_frame(node1, node2, "add")

    @classmethod
    def mul(cls, node1, node2):
        return Graph.binary_function_frame(node1, node2, "mul")

    @classmethod
    def div(cls, node1, node2):
        return Graph.binary_function_frame(node1, node2, "add")

    @classmethod
    def sub(cls, node1, node2):
        return Graph.binary_function_frame(node1, node2, "sub")

    @classmethod
    def pow(cls, node1, node2):
        return Graph.binary_function_frame(node1, node2, "pow")

    @classmethod
    def log(cls, node1):
        return Graph.unary_function_frame(node1, "log")

    @classmethod
    def sin(cls, node1):
        return Graph.unary_function_frame(node1, "sin")

    @classmethod
    def cos(cls, node1):
        return Graph.unary_function_frame(node1, "cos")


if __name__ == "__main__":
    """ Optimize a unary function """
    def f1(x):
        return ((x - 7) ** 2 + 10).log()
    x = Graph.Node(6, requires_grad=True)
    lr = 0.01
    history = []
    for i in range(2000):
        y = f1(x)
        history.append(y.value)
        y.backward()
        x.value = x.value - x.grad * lr
        Graph.kill_grad()
    plt.plot(history)
    plt.show()
    Graph.clear()

    """ Optimize a binary function """
    def f2(x, y):
        return 0.5 * x ** 2 + x * y + 0.5 * y ** 2 - 2 * x - 2 * y
    x = Graph.Node(6, requires_grad=True)
    y = Graph.Node(6, requires_grad=True)
    lr = 0.01
    history = []
    for i in range(1000):
        z = f2(x, y)
        history.append(z.value)
        z.backward()
        x.value = x.value - x.grad * lr
        y.value = y.value - y.grad * lr
        Graph.kill_grad()
    plt.plot(history)
    plt.show()
    Graph.clear()








