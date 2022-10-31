"""
在写完operator.py后，发现每次构建一张图都要定义变量（这是我们可以接受的），
但还要定义operator的节点和手动构建有向边（这是我们不想要的），也是用户不友好的。
我们希望能够通过 x=Variable(), y=Variable(), z=x+y就能自动实现图的构建。
因此在operator.template.py里面实现一些函数模板，来封装构建图的操作。
最后将这些函数模板作为classmethod放在Graph类下。
即在graph.py中继承naive_graph.py中的naive_Graph类.
"""

from operators import *
from node import *


def unary_function_frame_s(node, operator):   # example: node = Variable(3)   operator='exp'
    """
    一元函数：
    input: x, f
    return: f(x)
    """
    if not isinstance(node, naive_Graph.Node):    # 如果node和Graph.Node不是同一类或者不是它的子类，而是一个数
        node = Constant(node)               # 则先转化成Constant
    node_operator = Operator(operator)      # 节点 operator
    node.build_edge(node_operator)          # node → node_operator
    return node_operator


def binary_function_frame_s(node1, node2, operator):
    """
    二元函数
    input: x1, x2, f
    return: f(x1, x2)
    """
    if not isinstance(node1, naive_Graph.Node):
        node1 = Constant(node1)
    if not isinstance(node2, naive_Graph.Node):
        node2 = Constant(node2)
    node_operator = Operator(operator)
    node1.build_edge(node_operator)
    node2.build_edge(node_operator)
    return node_operator

# 上面的二元函数中，如果node1是具有"add"的operator，且当前operator也是”add“，
# 则会再创建一个"add"的operator，被node1和node2指向。
# 但这时最省事的做法应该是直接将node2指向node1.
# 上述这种省事的做法可以在连加和连乘这种满足结合律的operator的情况中使用
# 因此重写一个commutable_binary_function_frame函数来考虑满足结合律的operator的情况。


def commutable_binary_function_frame_s(node1, node2, operator):
    """
    A more efficient implementation of "binary_function_frame()".
    Constraint: operator is required to meet the associative law.
    """
    if not isinstance(node1, naive_Graph.Node):
        node1 = Constant(node1)
    if not isinstance(node2, naive_Graph.Node):
        node2 = Constant(node2)

    if isinstance(node1, Operator) and node1.operator == operator:
        node2.build_edge(node1)
        return node1
    elif isinstance(node2, Operator) and node2.operator == operator:
        node1.build_edge(node2)
        return node2
    else:
        node_operator = Operator(operator)
        node1.build_edge(node_operator)
        node2.build_edge(node_operator)
        return node_operator

# 最后将这三个函数模板作为三个类方法添加到graph.py的Graph类下面

