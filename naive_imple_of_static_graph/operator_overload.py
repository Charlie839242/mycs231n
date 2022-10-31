"""
这里我们将以例子来描述如何通过简单的：
x = Variable(1)
y = Variable(2)
s = x + y
来实现整个图的构造。
"""


"""
在写完operator_template.py后就有了三个模板函数，但不能让用户根据自己所采用operator来选择需要的函数模板。
我们需要针对具体不同的operator（’add‘，’sub‘，······）继续进行封装，
也就是要预先决定每个operator采用哪个函数模板，并封装导operator层。
例如，定义add函数: return Graph.commutable_binary_function_frame(node1, node2, "add")
这样用户接触到的就是add函数而非commutable_binary_function_frame函数
这些函数应该是naive_Graph的类方法
"""
from operators import *


@classmethod
def add(cls, node1, node2):
    return naive_Graph.commutable_binary_function_frame(node1, node2, "add")


@classmethod
def sub(cls, node1, node2):
    return naive_Graph.binary_function_frame(node1, node2, "sub")


@classmethod
def exp(cls, node1):
    return naive_Graph.unary_function_frame(node1, "exp")


"""
接下来对Node针对不同的operator进行运算符重载
以使得再进行operator(如加减···)时能够直接调用上述封装的方法，
这样就自动实现了节点之间有向边的构建。
所以就是对Node类进行运算符重载,上面已经定义了Node，所以这里直接
继承并覆盖Node，再向其中添加运算符重载的类方法。
"""

class Node(naive_Graph.Node):
    def __init__(self):
        super().__init__()

    def __add__(self, node):  # 重构加法
        return naive_Graph.add(self, node)

    def __sub__(self, node):  # 重构减法
        return naive_Graph.sub(self, node)

    def __pow__(self):  # 重构指数
        return naive_Graph.exp(self)