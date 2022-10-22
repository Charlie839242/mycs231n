from node import *

from math import prod
from math import exp as math_exp, log as math_log
from math import sin as math_sin, cos as math_cos
"""
在node.py里定义的三种变量是相互独立的，也就是说它们之间不能直接进行相互运算。
例如，不能直接Variable() + Constant()
因此定义了Operator类来实现变量之间的运算。
因为Constant只有value没有grad，PlaceHolder的value取决于输入，Variable既有value又有grad。
所以让Operator继承自Variable。
"""

# 为了方便起见，在这个系统里我们只实现以下8种运算的前向和后向。
operator_calculate_table = {
    # operator_calculate_table针对不同运算，给定一个节点a，
    # 可以根据与a相连的上游节点（node.last）来计算节点计算的结果。
    "add": lambda node: sum([last.get_value() for last in node.last]),
    "mul": lambda node: prod([last.get_value() for last in node.last]),
    "div":
    lambda node: node.last[0].get_value() / node.last[1].get_value(),
    "sub":
    lambda node: node.last[0].get_value() - node.last[1].get_value(),
    "exp": lambda node: math_exp(node.last[0].get_value()),
    "log": lambda node: math_log(node.last[0].get_value()),
    "sin": lambda node: math_sin(node.last[0].get_value()),
    "cos": lambda node: math_cos(node.last[0].get_value()),
}


class Operator(Variable):
    def __init__(self, operator: str):
        super().__init__(0)
        self.operator = operator    # example:  self.operator = 'add'
        # self.calculate保存了该节点前向传播的lambda表达式
        self.calculate = operator_calculate_table[operator]  # example: operator.calculate(Node)

    def __repr__(self) -> str:
        return self.operator


if __name__ == "__main__":
    """  二元计算: x + y: 这里我们只建图，计算的过程交给后面的前向传播来完成  """
    x = Constant(1)             # 变量 x
    y = Variable(2)             # 变量 y
    add = Operator('add')       # 节点 add
    x.build_edge(add)           # 构建 x → add 的有向边
    y.build_edge(add)           # 构建 y → add 的有向边
    print(x.next)
    print(y.next)

    """  一元计算: e^z  """
    z = Variable(3)             # 变量 z
    exp = Operator('exp')       # 节点 exp
    z.build_edge(exp)           # 构建 z → exp 的有向边
    print(z.next)



