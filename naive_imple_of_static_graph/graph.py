"""
这里将naive_graph.py, node.py, operator.py, operator_template.py and operator_overload.py
的内容融合在一起。以实现：
x = Variable(1)
y = Variable(2)
s = x + y
就能建图的效果。
之所以要放在一个文件里，是因为node引用了naive_graph, operator引用了node, operator_template引用了
operator, 而naive_graph又要引用operator_template, 这里分开写存在circular import的问题。
"""
import random
import matplotlib.pyplot as plt
from random import randint
from math import prod, e
from math import log as math_log, pow as math_pow
from math import sin as math_sin, cos as math_cos


class Graph:
    """
    Define the class for the computational graph
    and the node in computational graph.
    """
    # 全局只有一个计算图，即node_list和id_list是全局唯一的
    # 且这样写可以不用实例化naive_Graph而操作和查看node_list和id_list
    node_list = []  # 图节点列表
    id_list = []    # 节点ID列表
    operator_calculate_table = {
        # operator_calculate_table针对不同运算，给定一个节点a，
        # 可以根据与a相连的上游节点（node.last）来计算节点计算的结果。
        "add": lambda node: sum([last.get_value() for last in node.last]),
        "mul": lambda node: prod([last.get_value() for last in node.last]),
        "div": lambda node: node.last[0].get_value() / node.last[1].get_value(),
        "sub": lambda node: node.last[0].get_value() - node.last[1].get_value(),
        "pow": lambda node: math_pow(node.last[0].get_value(), node.last[1].get_value()),
        # "exp": lambda node: math_exp(node.last[0].get_value()),
        "log": lambda node: math_log(node.last[0].get_value()),
        "sin": lambda node: math_sin(node.last[0].get_value()),
        "cos": lambda node: math_cos(node.last[0].get_value()),
    }

    class Node:
        def __init__(self):
            """
            Initialize Graph.Node is equal to create a new node that
            hasn't been connected to anything and add it into the Graph.
            """
            self.next = list()  # 与self直接相连的下游节点
            self.last = list()  # 与self直接相连的上游节点
            self.in_deg, self.in_deg_com = 0, 0     # 节点入度
            self.out_deg, self.out_deg_com = 0, 0   # 节点出度
            Graph.add_node(self)

        def build_edge(self, node):
            # 构建self指向node的有向边
            self.out_deg += 1
            node.in_deg += 1
            self.next.append(node)
            node.last.append(self)

        """ 重构Node运算符 """
        def __add__(self, node):    # 重构加法: self出现在加号左边
            return Graph.add(self, node)

        # def __radd__(self, node):   # 重构加法: self出现在加号右边
        #     return Graph.add(node, self)

        def __mul__(self, node):    # 重构乘法
            return Graph.mul(self, node)

        def __truediv__(self, node):    # 重构除法
            return Graph.div(self, node)

        def __sub__(self, node):    # 重构减法
            return Graph.sub(self, node)

        def __pow__(self, node):
            return Graph.pow(self, node)

        def __rpow__(self, node):  # 重构指数
            return Graph.pow(node, self)

        """ 
        接下来还需要实现log, sin, cos的重载 
        但这些运算没有相应自带的重载运算符，所以以类方法的形式来实现。
        """
        def sin(self):
            return Graph.sin(self)

        def cos(self):
            return Graph.cos(self)

        def log(self):
            return Graph.log(self)

    @classmethod
    def add_node(cls, node):
        # 在计算图中加入节点
        cls.node_list.append(node)

    @classmethod
    def clear(cls):
        # 清空计算图
        cls.node_list.clear()

    @classmethod
    def kill_grad(cls):
        # 清空所有梯度
        for node in cls.node_list:
            node.grad = 0

    """
    node.py的内容
    """

    class Constant(Node):
        def __init__(self, value):
            super().__init__()
            self.value = float(value)

        def get_value(self):
            return self.value

        def __repr__(self):
            return str(self.value)

    class Variable(Node):
        def __init__(self, value):
            super().__init__()
            self.value = float(value)
            self.grad = 0.

        def get_value(self):
            return self.value

        def __repr__(self):
            return str(self.value)

    class PlaceHolder(Node):
        def __init__(self):
            super().__init__()
            self.value = None
            self.grad = 0.

        def get_value(self):
            return self.value

        def __repr__(self):
            return str(self.value)

    """
    operator.py的内容
    """
    class Operator(Variable):
        def __init__(self, operator: str):
            super().__init__(0)
            self.operator = operator  # example:  self.operator = 'add'
            # self.calculate保存了该节点前向传播的lambda表达式
            self.calculate = Graph.operator_calculate_table[operator]  # example: operator.calculate(Node)

        def __repr__(self) -> str:
            return self.operator

    """
    operator_template.py的内容
    """
    @classmethod
    def unary_function_frame(cls, node, operator):
        if not isinstance(node, Graph.Node):  # 如果node和Graph.Node不是同一类或者不是它的子类，而是一个数
            node = Graph.Constant(node)  # 则先转化成Constant
        node_operator = Graph.Operator(operator)  # 节点 operator
        node.build_edge(node_operator)  # node → node_operator
        return node_operator

    @classmethod
    def binary_function_frame(cls, node1, node2, operator):
        if not isinstance(node1, Graph.Node):
            node1 = Graph.Constant(node1)
        if not isinstance(node2, Graph.Node):
            node2 = Graph.Constant(node2)
        node_operator = Graph.Operator(operator)
        node1.build_edge(node_operator)
        node2.build_edge(node_operator)
        return node_operator

    @classmethod
    def commutable_binary_function_frame(cls, node1, node2, operator):
        if not isinstance(node1, Graph.Node):
            node1 = Graph.Constant(node1)
        if not isinstance(node2, Graph.Node):
            node2 = Graph.Constant(node2)

        if isinstance(node1, Graph.Operator) and node1.operator == operator:
            node2.build_edge(node1)
            return node1
        elif isinstance(node2, Graph.Operator) and node2.operator == operator:
            node1.build_edge(node2)
            return node2
        else:
            node_operator = Graph.Operator(operator)
            node1.build_edge(node_operator)
            node2.build_edge(node_operator)
            return node_operator

    """
    有了三个模板函数后，不能让用户根据自己所采用operator来选择需要的函数模板。
    我们需要针对具体不同的operator（’add‘，’sub‘，······）继续进行封装，
    也就是要预先决定每个operator采用哪个函数模板，并封装导operator层。
    例如，定义add函数: return Graph.commutable_binary_function_frame(node1, node2, "add")
    这样用户接触到的就是add函数而非commutable_binary_function_frame函数
    所以这里我们为八种运算分配相应的函数模板
    """
    @classmethod
    def add(cls, node1, node2):
        return Graph.commutable_binary_function_frame(node1, node2, "add")

    @classmethod
    def mul(cls, node1, node2):
        return Graph.commutable_binary_function_frame(node1, node2, "mul")

    @classmethod
    def div(cls, node1, node2):
        return Graph.commutable_binary_function_frame(node1, node2, "add")

    @classmethod
    def sub(cls, node1, node2):
        return Graph.binary_function_frame(node1, node2, "sub")

    @classmethod
    def pow(cls, node1, node2):
        return Graph.binary_function_frame(node1, node2, "pow")

    @classmethod
    def exp(cls, node1):
        return Graph.unary_function_frame(node1, "exp")

    @classmethod
    def log(cls, node1):
        return Graph.unary_function_frame(node1, "log")

    @classmethod
    def sin(cls, node1):
        return Graph.unary_function_frame(node1, "sin")

    @classmethod
    def cos(cls, node1):
        return Graph.unary_function_frame(node1, "cos")

    """
    接下来对Node针对不同的operator进行运算符重载
    以使得再进行operator(如加减···)时能够直接调用上述封装的方法，
    这样就自动实现了节点之间有向边的构建。
    所以就是对Node类进行运算符重载,直接回到之前的Node类中添加新的方法
    """

    """
    在Node类中对八种运算重载完成后，就意味着图构建完成了。
    然后我们需要对八种运算分别写出其导数
    然后封装在查询函数__deriv中
    """

    @classmethod
    def __deriv_add(cls, child, parent):
        return 1

    @classmethod
    def __deriv_sub(cls, child, parent):
        if child.last[0] == parent:
            return 1
        elif child.last[1] == parent:
            return -1

    @classmethod
    def __deriv_mul(cls, child, parent):
        if child.last[0] == parent:
            return child.last[1].value
        elif child.last[1] == parent:
            return child.last[0].value

    @classmethod
    def __deriv_div(cls, child, parent):
        if child.last[0] == parent:
            return 1 / child.last[1].value
        elif child.last[1] == parent:
            return -child.last[0].value/(child.last[1].value**2)

    @classmethod
    def __deriv_pow(cls, child, parent):
        if child.last[0] == parent:
            return child.last[1].value * math_pow(child.last[0].value, child.last[1].value - 1)
        elif child.last[1] == parent:
            return child.value * math_log(child.last[0].value)

    @classmethod
    def __deriv_log(cls, child, parent):
        return 1 / parent.value

    @classmethod
    def __deriv_sin(cls, child, parent):
        return math_cos(parent.value)

    @classmethod
    def __deriv_cos(cls, child, parent):
        return -math_sin(parent.value)

    @classmethod
    def __deriv(cls, child: Operator, parent: Node):
        return {
            "add": cls.__deriv_add,
            "sub": cls.__deriv_sub,
            "mul": cls.__deriv_mul,
            "div": cls.__deriv_div,
            "pow": cls.__deriv_pow,
            "log": cls.__deriv_log,
            "sin": cls.__deriv_sin,
            "cos": cls.__deriv_cos
        }[child.operator](child, parent)

    """
    下一步我们需要完成前向传播，也就是对图先进行拓扑排序。
    """

    @classmethod
    def forward(cls):
        """
        一个原则：如果一个节点的入度是0，那么它就应该是求好值的。
        首先删除所有入度为0的节点，并修改这些节点的下游节点的入度。
        若写有节点的入度变成了0，则计算它的值，并将该节点继续删除
        然后修改其下游节点的入度。
        重复上述过程。
        """
        node_queue = []  # 节点队列

        # 入度为0的节点添加进node_queue
        for node in cls.node_list:
            if node.in_deg == 0:
                node_queue.append(node)

        while len(node_queue) > 0:
            node = node_queue.pop()
            # 依次删除每个入度为0的节点，并更新其下游节点的入度
            for next_node in node.next:
                next_node.in_deg -= 1
                next_node.in_deg_com += 1
                # 若下游节点入度是0，则计算其值并删除该节点，重复上述过程
                if next_node.in_deg == 0:
                    next_node.value = next_node.calculate(next_node)
                    node_queue.insert(0, next_node)

        # 通过in_deg_com来还原in_deg
        for node in cls.node_list:
            node.in_deg += node.in_deg_com
            node.in_deg_com = 0

    """ 
    然后是反向传播
    反向传播就是逆拓扑排序 
    """

    @classmethod
    def backward(cls, y=None):
        if y == None:
            """ Backward on the whole graph """
            node_queue = []
            for node in cls.node_list:
                # 找到所有出度为0的节点且不是Constant
                if node.out_deg == 0 and not isinstance(node, Graph.Constant):
                    node.grad = 1.
                    node_queue.append(node)
            if len(node_queue) > 1:
                """
                这种做法在有多个输出时是不太合理的
                例如有两个输出f1和f2，输入x
                最后求得关于x的导数是df1/dx + df2/dx
                而我们希望得到的是二者分别的值df1/dx, df2/dx
                """
                pass
        else:
            """ Backward only from the node y """
            assert type(y) != cls.Constant, "常量无法求导"
            node_queue = []
            for node in cls.node_list:
                if node.out_deg == 0 and not isinstance(node, cls.Constant):
                    if node == y:
                        node.grad = 1.
                    else:
                        node.grad = 0.
                    node_queue.append(node)

        while len(node_queue) > 0:
            node = node_queue.pop()
            # 依次删除每个出度=0的节点，并更新其上游节点的出度
            for last_node in node.last:
                last_node.out_deg -= 1
                last_node.out_deg_com += 1
                # 如果上游节点的出度变成了0，即可以求到这个节点的导数，并将其删除，重复上述过程
                if last_node.out_deg == 0 and not isinstance(last_node, cls.Constant):
                    # last_node的梯度由其所有出度传回来的梯度和
                    for n in last_node.next:
                        last_node.grad += n.grad * cls.__deriv(n, last_node)
                    # 下个循环将继续删除该出度为0的节点
                    node_queue.insert(0, last_node)

        # 恢复所有节点的出度
        for node in cls.node_list:
            node.out_deg += node.out_deg_com
            node.out_deg_com = 0

    @classmethod
    def backward_from_node(cls, y):
        assert type(y) != cls.Constant, "常量无法求导"

        node_queue = []
        for node in cls.node_list:
            if node.out_deg == 0 and not isinstance(
                    node,
                    cls.Constant,
            ):
                if node == y:
                    node.grad = 1.
                else:
                    node.grad = 0.
                node_queue.append(node)

        while len(node_queue) > 0:
            node = node_queue.pop()
            # 依次删除每个出度=0的节点，并更新其上游节点的出度
            for last_node in node.last:
                last_node.out_deg -= 1
                last_node.out_deg_com += 1
                # 如果上游节点的出度变成了0，即可以求到这个节点的导数，并将其删除，重复上述过程
                if last_node.out_deg == 0 and not isinstance(last_node, Graph.Constant):
                    # last_node的梯度由其所有出度传回来的梯度和
                    for n in last_node.next:
                        last_node.grad += n.grad * cls.__deriv(n, last_node)
                    # 下个循环将继续删除该出度为0的节点
                    node_queue.insert(0, last_node)

            # 恢复所有节点的出度
        for node in cls.node_list:
            node.out_deg += node.out_deg_com
            node.out_deg_com = 0


if __name__ == "__main__":
    """ Example: Addition """
    x = Graph.Variable(1)
    y = Graph.Variable(2)
    z = Graph.Variable(3)
    s = x + y + z
    print(s.in_deg)
    print(x.next)
    print(y.next)
    print(z.next)
    print(Graph.node_list)
    print(Graph.node_list[3].last)
    Graph.clear()

    """ Example: Exp"""
    a = Graph.Variable(4)
    m = e**a
    print(Graph.node_list)
    print(a.next)
    print(Graph.node_list[1].last)
    Graph.clear()

    """ Example: Sin """
    q = Graph.Variable(5)
    t = q.sin()
    r = t.cos()
    print(Graph.node_list)
    print(r.last)
    print(t.last)
    Graph.clear()

    """ Example: Forward pass """
    x = Graph.Variable(1)
    y = Graph.Variable(2)
    z = Graph.Variable(3)
    s = x - y.sin() + e**z
    Graph.forward()
    print(s.value)
    print(x.next, y.next, z.next)
    print(s)

    """ Example: Backward pass """
    Graph.backward()
    print(Graph.node_list)
    print(x.grad)
    print(y.grad)
    print(z.grad)
    Graph.kill_grad()

    """ Example: Backward pass from node """
    Graph.backward(s)
    print(Graph.node_list)
    print(x.grad)
    print(y.grad)
    print(z.grad)
    Graph.kill_grad()
    Graph.clear()

    """ Example: Find the lowest value for function f """
    x = Graph.Variable(6)       # x初始值为3
    y = Graph.Constant(7)
    z = Graph.Constant(10)
    f = ((x - y) ** 2 + 10).log()
    lr = 0.01      # random learning rate
    history = []
    for _ in range(2000):
        Graph.forward()
        Graph.backward(f)
        x.value = x.value - x.grad * lr
        print(x)
        Graph.kill_grad()
        history.append(f.value)
    plt.plot(history)
    plt.show()







