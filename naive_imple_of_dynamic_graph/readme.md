# Intro

之前我们实现了标量的静态计算图图构建，现在我们来实现动态计算图图的构建。

Tensorflow1采用的就是静态图，但是静态图有一个很反直觉的设定就是，调用计算函数后，用户无法得到计算的结果，因为这种计算函数的目的是建图，而不是计算。所有的计算必须要等到最后的前向传播才能进行，以之前的代码为例：

```Python
x = Graph.Variable(1)
y = Graph.Variable(2)
z = x + y
print(z.value)------>0.0
Graph.forward()
print(z.value)------>3.0
```

可以看见，在静态计算图中必须进行前向传播后才能得到z的值。

然而我们希望z = x + y后能够马上得到z的值，这就是动态图的思想。因此，动态图和静态图的一个差别在于：

- 静态图：1次建图 + n次（前向传播 + 反向传播）。
- 动态图： n次（建图/前向传播 + 反向传播），动态图中建图和前向传播同时实现。

考虑动态图的上述特性，因为涉及到了n次建图，如果不采取措施的话，势必会产生许多冗余节点，因此动态图的一个重要设定是：**在反向传播后销毁计算图**。如何删除将在下面介绍。

还有一个非常重要的点，静态图的前向传播和反向传播要在一张图上进行，所以静态图必须包含前向传播和反向传播涉及到的所有节点。但动态图不一样，动态图的建图和前向传播是同时进行的（可以近似认为没有前向传播），所以只有反向传播在动态图上进行，所以构建动态图时只需要向其中添加反向传播需要的节点。



# How does Pytorch do

Pytorch里面采用的就是动态图，我们先来看看它里面的tensor是如何设计的，以借鉴到我们自己的系统里。

1. 首先执行以下代码。

   ```Python
   import torch
   x = torch.ones((1, 1), requires_grad=True)
   y = torch.rand((1, 1))
   z = x + y
   z.backward()
   print(x.requires_grad, x.grad)--->True tensor([[1.]])
   print(y.requires_grad, y.grad)--->False None
   print(z.requires_grad, z.grad)--->True None
   ```

   上述结果说明了两件事：

   - 一个tensor的requires_grad=True，则任何依赖于该tensor的tensor的requires_grad自动为True
   - y.grad=None说明了requires_grad为False的tensor在反向传播中不会计算梯度。

2. 然而还有一个问题，为什么z.requires_grad=True，然而反向传播后z的梯度却是None？

   这个问题可以从tensor的分类说起。所有tensor可以分作两类，叶节点（leaf tensor）和非叶节点（computed tensor），查看是否是叶节点可以从tensor.is_leaf判断。引入叶节点的初心是因为：保留所有requires_grad=True的tensor的梯度太浪费内存了，能不能只保留叶节点的梯度呢？什么样的tensor才能成为叶节点呢？

   成为叶节点的条件：该tensor不依赖于其他任何tensor。也就是说：

   - 对于用户自己定义的tensor，不论其requires_grad是True还是False，一定是叶节点，因为用户自己定义意味着该tensor不是任何operation的结果。
   - 所有requires_grad=False的tensor，一定是叶节点。（这一归类实际没有意义，因为只有叶节点的梯度会被保留下来，但requires_grad=False导致对该tensor根本就不会求导，因此无论如何requires_grad=False的tensor都是没有梯度的。但这样处理能方便归类，见下）

   上面的描述比较口语，可以引入tensor.grad_fn，tensor.grad_fn储存了获取该tensor的方式并将用于反向传播。

   - requires_grad=False的tensor一定是叶节点，此时tensor.grad_fn=None（requires_grad=False的tensor根本没有梯度流过，所以其grad_fn=None是合理的）。
   - requires_grad=True的tensor既有可能是叶节点，也有可能非叶节点。此时看tensor.grad_fn，如果是None（用户自己创建的），则是叶节点，否则（通过计算得到的）为非叶节点。

   可以看见，如果将requires_grad=False的tensor归为叶节点，则有结论：grad_fn为None则一定是叶节点，反之则是非叶节点。这就是将requires_grad=False的tensor归为叶节点的好处。

   讨论了这么多，总结一下，确定一个节点的属性的步骤：

   - 先确定requires_grad，默认为False，由用户传入的参数或者上游节点的属性决定。requires_grad=True的节点才会进入动态图，requires_grad=False的节点不会进入动态图。
   - 再确定grad_fn。requires_grad=False的节点的grad_fn一定为None；requires_grad = True的节点，若是用户创建的，则grad_fn = None；若是由计算得到的，则grad_fn ≠ None。
   - 最后确定is_leaf。grad_fn为None则为叶节点，反之则为非叶节点。

   现在解释一开始的问题就很容易了，因为z是非叶节点，所以z的梯度会被计算，但在最后不会被保存下来。

3. 在了解了叶节点和非叶节点后，那么“销毁计算图”这一过程就可以实现：
   - 删除所有节点的上游节点和下游节点信息。
   - 然后只保留叶节点。
   - 最终保留的节点都是孤立节点。



# Graph and Node

相似的，这里先定义动态图和图中节点的基本结构，同样利用嵌套类的方法。

- 动态图比静态图多出一个销毁计算图的过程。销毁计算图的过程就是删除所有节点的上下游信息，并只保留叶节点。
- 动态图中没有专门的前向传播过程，所以动态图中节点不需要入度属性。
- 动态图仍然会涉及到在同一张图上作反向传播（retain_graph=True），所以需要清空梯度的类方法。
- 每当一个节点创建后，只有当其requires_grad=True时才会自动添加进计算图，反之不会添加进计算图。

```Python
class Graph:
    # node_list只保存计算图里的节点(也就是反向传播需要的节点)
    node_list = []

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
            
```

这里我们引入了requires_grad属性来显式的表明一个Node是否需要求梯度等等，所以我们不在需要像静态图中将Node分为Constant、Variable、Placeholder和Operator来为每个Node分配对应工作，而是可以将所有变量统一为Node类。

考虑一个Node被实例化时的情况，有两种情况：用户定义和计算得到（计算得到即是通过函数模板得到，函数模板定义在下一部分）。

- 用户定义：此时requires_grad由用户自己决定，grad_fn默认为None，因此is_leaf=True，该节点成为叶节点。
- 计算得到：此时requires_grad由上游节点决定：
  - requires_grad=True：则grad_fn=求导函数，is_leaf=False
  - requires_grad=False：则grad_fn=None，is_leaf=True



# Calculation between Nodes

同样的，我们采用函数模板来实现Node之间的计算。

- 函数模板需要接收node和operation（可以是字符串），来生成node经过operation后的新node。
- 例如执行z=x+y，我们需要在生成z节点的同时使得z的值已经计算出来了，也就是建图和前向传播同时进行。
- 生成的new node的requires_grad属性取决于其上游节点，并且requires_grad属性也会决定new node是否加入计算图。
- grad_fn在节点创建时被赋值。新node只有在其requires_grad=True的时候grad_fn≠None，requires_grad=False时没有梯度流过该节点。

```Python
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
```

grad_fn_table是一个字典，包含了各个算子的求值函数和求导函数。将在下一部分介绍。



# Operators and Derivatives

grad_fn_table是一个字典，包含了各个算子的求值函数和求导函数。通过：

```Python
fn, grad_fn = cls.grad_fn_table.get(operator)
```

来获取某个operator的前向过程和反向过程的匿名函数。



```Python
from math import prod, e
from math import log as math_log, pow as math_pow
from math import sin as math_sin, cos as math_cos

grad_fn_table = {
    "add": (forward_calculate_table.get('add'), backward_calculate_table('add')),
    "mul": (forward_calculate_table.get('mul'), backward_calculate_table('mul')),
    "div": (forward_calculate_table.get('div'), backward_calculate_table('div')),
    "sub": (forward_calculate_table.get('sub'), backward_calculate_table('sub')),
    "pow": (forward_calculate_table.get('pow'), backward_calculate_table('pow')),
    "log": (forward_calculate_table.get('log'), backward_calculate_table('log')),
    "sin": (forward_calculate_table.get('sin'), backward_calculate_table('sin')),
    "cos": (forward_calculate_table.get('cos'), backward_calculate_table('cos')),
}
```

其中forward_calculate_table就是各个算子的求值函数组成的字典，backward_calculate_table包含各个算子的求导函数组成的字典。



首先是求值函数，通过字符串-函数的字典储存：

```Python
forward_calculate_table = {
    "add": lambda node1, node2: node1.value + node2.value,
    "sub": lambda node1, node2: node1.value - node2.value,
    "div": lambda node1, node2: node1.value / node2.value,
    "pow": lambda node1, node2: math_pow(node1.value, node2.value),
    "mul": lambda node1, node2: node1.value * node2.value,
    "log": lambda node1: math_log(node1.value),
    "sin": lambda node1: math_sin(node1.value),
    "cos": lambda node1: math_cos(node1.value)
}
```



然后是求导函数，因为求导函数要进行判断等等操作，单独的lambda储存不下，所以采用lambda和函数嵌套的方法：

```
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
        return -child.last[0].value/(child.last[1].value**2)

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
```



# Overload Operator

上一部分已经定义了函数模板以及写出了求值和求导函数。

然后根据写出的一元、二元函数模板来定义八种算子的操作（即每种算子用哪个模板）：

```Python
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
```



接下来需要重载node的操作符：

```Python
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

```



# BackPropagation

回忆我们在静态图里是怎么做的反向传播：

```Python
Graph.backward(y)
```

这里Graph.backward(y)大致的过程是从整个Graph的角度，找到所有出度为0的node（应该包含y），然后将除了y之外所有出度为0的node的梯度置0，y的梯度置1。然后作逆拓扑排序并用求导函数来传播梯度。

这里我们希望通过：

```Python
y.backward()
```

来实现从y开始的反向传播过程。因此，backward()应该是Node类的一个类方法，具体方法和静态图中反向传播类似，都是对有向无环图作逆拓扑排序，稍有不同的是在传播完毕后是选择销毁计算图还是保留计算图。

设计参数retain_graph，retain_graph=True时会保留计算图，这种情况一般用于有多个输出，要对多个输出）（例如loss1和loss2）同时求导时，先对loss1求导，若此时销毁计算图，则无法再对loss2求导；若选择保留计算图，则可以清空梯度后再对loss2求导。

```Python
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
        	# 从图中删除node
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
		Graph.free_graph()
```



# Overload Operator

然后根据写出的一元、二元函数模板来定义八种算子的操作：

```Python
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
```



接下来需要重载node的操作符：

```Python
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

```

至此，所有上述内容保存在graph.py里。



# Example

对函数$f(x)=log((x-7)^2+10)$找到最小值。

```Python
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
```

优化过程如下：

![image](https://github.com/Charlie839242/mycs231n/blob/main/naive_imple_of_dynamic_graph/img/Figure_1.png)  



对函数$f(x,y)=\frac{1}{2}x^2+xy+\frac{1}{2}y^2-2x-2y$找到最小值。

```Python
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
```

优化过程如下：

![image](https://github.com/Charlie839242/mycs231n/blob/main/naive_imple_of_dynamic_graph/img/Figure_2.png)  





















































