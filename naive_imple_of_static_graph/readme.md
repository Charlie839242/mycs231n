# Intro

这里我们参考计算图（静态图）的思想，来构造一个非常naive的计算图框架。非常naive是指计算图中没有矩阵参与运算，而都是浮点数在运算。对这个计算图框架，我们希望它具有：

- 具有计算图和节点的基本数据结构，且计算图是全局的，每创建一个节点会自动添加到当前计算图里，也就是说像下面这样就能自动构建计算图。

  ```python
  x = Graph.Variable(1)
  y = Graph.Constant(2)
  z = x + y
  ```

- 具有静态图的特性，即先建图再计算。也就是说上述代码的作用只是创建一张计算图，在执行完上述代码后用户仍无法得知z的值，需要格外的函数来执行前向传播（获取z的值）和反向传播（更新权重）。

- 能够进行前向传播和反向传播。



# Reading Sequence：

naive_graph.py → node.py → operators.py → operator_template.py → operator_overload.py → graph.py



# naive_graph.py

在这里，我们首先抽象出计算图（Computational Graph）以及构成计算图的节点（Node）并编写二者的class。

一个计算图本质上是一个有向无环图，因此计算图既需要保存其拥有的节点，还需要储存每个节点与其他节点之间的有向关系。基于此，我们令计算图只用于保存位于其中的所有节点，各个节点之间的关系我们则希望能体现在节点class的属性中，一个方法便是对于节点class，定义其上游节点属性和下游节点属性，这样对于每个节点，我们就能通过其上游节点和下游节点来判断它们之间的有向关系。思考二者分别应该具有的属性。

计算图：

- 一个node_list（list），来存放所有处于该计算图中的节点。

- 一系列类方法，对该计算图中的节点（node_list）进行处理。

  例如，一个add_node的类方法，用于将节点添加进计算图中。

节点：

- 上游节点属性和下游节点属性，以储存节点之间的有向关系。

- 节点的入度、出度属性，以代表是否有其他节点（被）指向该节点。

- 一系列类方法，以修改上述两类属性。

  例如一个build_edge(a, b)的类方法，可以让a节点指向b。那么该类方法中就要赋值b是a的下游节点，a是b的下游节点，且a的出度+1，b的入度+1。

因此，计算图和节点的类定义如下：

```python
class naive_Graph:
    """
    Define the class for the computational graph
    and the node in computational graph.
    """
    node_list = []  # 图节点列表

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
            naive_Graph.add_node(self)

        def build_edge(self, node):
            # 构建self指向node的有向边
            self.out_deg += 1
            node.in_deg += 1
            self.next.append(node)
            node.last.append(self)

    @classmethod
    def add_node(cls, node):
        # 在计算图中加入节点
        cls.node_list.append(node)

    @classmethod
    def clear(cls):
        # 清空计算图
        cls.node_list.clear()
```

- 这样定义node_list而不用def \_\_init\_\_的一大好处是node_list是全局共享的，比如说：

  ```python
  graph1 = naive_Graph()
  graph2 = naive_Graph()
  graph1.node_list.append(1)
  print('graph2:', graph2.node_list)
  -----------------------
  graph2:[1]
  ```

  另一好处是node_list不用实例化naive_Graph就可以通过naive_Graph.node_list直接调用。

- 将Node的类设计为naive_Graph类里的嵌套类，这样的好处在于，Node里的类方法里可以直接调用naive_Graph的类方法。比如说Node类在 \_\_init\_\_中最后调用了naive_Graph.add_node(self)来将自身添加进计算图中。因此，每当我们创建一个节点，该节点就会通过\_\_init\_\_自动添加进计算图中，从而省去了将节点显式加入计算图这一过程。



# node.py

在上面的过程中，我们定义了计算图和计算图节点的基本结构。

但是像上述的节点是不够的，它还需要有值，梯度等属性。因此，我们通过继承node类，构建了三种变量：常量Constant、变量Variable、占位符Placeholder。

- Constant：不存在梯度，只存在值。

- Variable：既有梯度，也有值。

- Placeholder：通常作为神经网络的数据输入口，有值，也有梯度。但在初始化时没有值。

  占位符是非常必要的，因为每增加一个Constant/Variable，计算图都会增加一个节点。而训练时每一次就会有一个输入，多个epoches会导致节点太多。

```Python
class Constant(naive_Graph.Node):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def get_value(self):
        return self.value

    def __repr__(self):
        return str(self.value)


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
```



# operator.py

上面定义的三种数据Constant、Variable和Placeholder是独立的，相互之间还不能进行运算。因此我们定义Operator类（一个operator也是计算图中的一个节点）来实现三种数据的运算。Operator也是具有值和梯度的，但是Operator比起那三种数据结构要多出关于怎么计算指定Operator的信息，我们将这一部分信息储存在属性self.calculate里面。所以，Operator类可以继承自Variable。

一个简单的例子便是一个Constant：x和Variable：y指向一个加法Operator：z，z的上游节点就是x和y，我们需要z能够通过它自身的上游节点来计算x+y的值。

```Python
class Operator(Variable):
    def __init__(self, operator: str):
        super().__init__(0)
        self.operator = operator    # example:  self.operator = 'add'
        self.calculate = operator_calculate_table[operator]  

    def __repr__(self) -> str:
        return self.operator
```

通过self.operator='add'来指定这是一个加法节点或是其他。再通过匿名函数lambda储存该Operator对其上游节点的操作，这些操作储存在字典operator_calculate_table里，想要增加新的算子也可以直接在operator_calculate_table里修改，这里只实现了8种算子。

```
operator_calculate_table = {
"add": lambda node: sum([last.get_value() for last in node.last]),
"mul": lambda node: prod([last.get_value() for last in node.last]),
"div": lambda node: node.last[0].get_value() / node.last[1].get_value(),
"sub": lambda node: node.last[0].get_value() - node.last[1].get_value(),
"pow": lambda node: math_pow(node.last[0].get_value(), node.last[1].get_value()),
"log": lambda node: math_log(node.last[0].get_value()),
"sin": lambda node: math_sin(node.last[0].get_value()),
"cos": lambda node: math_cos(node.last[0].get_value()),
}
```

匿名函数中lambda node的node就是当前的Operator，通过node.last获取Operator的上游节点，再进行相应的运算。

此时，我们若想建立x+y的计算图，就已经可以实现了：

```Python
x = Constant(1)             # 变量 x
y = Variable(2)             # 变量 y
add = Operator('add')       # 节点 add
x.build_edge(add)           # 构建 x → add 的有向边
y.build_edge(add)           # 构建 y → add 的有向边
```

需要注意的一点是，现在add的值仍不知道，上述过程我们只是进行建图，计算的过程交给后面的前向传播来完成，这也符合静态图的思想。



# operator_template.py

完成operator.py后，我们发现每次构建一张图，我们都需要定义变量（这是我们可以接受的），但还要定义operator的节点以及手动构建有向边（这是我们不想要的）。

我们希望能够通过：

```Python
x=Variable() 
y=Variable() 
z=x+y
```

就能够实现计算图的构建。

因此在这里我们想要实现一些函数模板，来封装构建计算图的操作。

仍然以z=x+y为例，我们希望通过如下操作就能构建整个图：

```Python
x = Constant(1)             
y = Variable(2)  
operator = undefined_function(x, y, 'add')
```

可以看见，我们希望不用手动x.build_edge和y.build_edge来构建计算图，也就是将build_edge封装进undefined_function即可。

根据Operator的种类，可以分为一元函数算子和二元函数算子。

```Python
# 一元函数
def unary_function_frame(node, operator):   
    if not isinstance(node, naive_Graph.Node):    
        node = Constant(node)               # 若是一个数则先转化成Constant
    node_operator = Operator(operator)      # 节点 Operator
    node.build_edge(node_operator)          # node → node_operator
    return node_operator

# 二元函数
def binary_function_frame(node1, node2, operator):
    if not isinstance(node1, naive_Graph.Node):
        node1 = Constant(node1)
    if not isinstance(node2, naive_Graph.Node):
        node2 = Constant(node2)
    node_operator = Operator(operator)
    node1.build_edge(node_operator)
    node2.build_edge(node_operator)
    return node_operator
```

我们注意到了一个事实，在面对y=a+b+c的情况时，如果采用binary_function_frame里的方法，那么首先a+b会产生一个add节点，我们将其称为add_1，a和b指向add_1；此时再加上c，则会再产生一个add节点，我们将其称为add_2，add_1和c指向add_2，这样也能解决问题，但却不是最优解。此时的最优解应该是c直接指向add_1，避免了一个冗余节点add_2的产生。

可以显然看出，会发生这种情况的条件是算子满足结合律，比如说加法，(a+b)+c=a+(b+c)，则会出现这种情况。在我们实现的八种算子里面，add和mul都会出现这种情况。

因此，我们针对满足结合律的二元函数重新写一个函数模板。

```Python
def commutable_binary_function_frame(node1, node2, operator):
	"""Operator must be commutable!"""
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
```

将这三个函数放在naive_Graph类下面，此时我们构建z=x+y的过程变成了：

```Python
x = Constant(1)             
y = Variable(2)  
operator = commutable_binary_function_frame(x, y, 'add')
```



# operator_overload.py

上述定义了三个函数模板，但不能让用户根据自己所采用operator来选择需要的函数模板。我们需要针对具体不同的operator（’add‘，’sub‘，······）继续进行封装，也就是要预先决定每个operator采用哪个函数模板。

```
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
```

目前已经能够通过：

```Python
x = Constant(1)             
y = Variable(2)  
operator = naive_Graph.add(x, y)
```

来构建计算图，但仍然不够简洁。

但我们希望通过：

```Python
x=Variable() 
y=Variable() 
z=x+y
```

来构建计算图。这就需要对Node类的运算符重载，以使得x+y时'+'的运算符重载能够直接调用上述封装好的函数，如naive_Graph.add等等。

```Python
class Node:
    """ 重构Node运算符 """
    def __add__(self, node):    # 重构加法: self出现在加号左边
        return naive_Graph.add(self, node)

    def __mul__(self, node):    # 重构乘法
        return naive_Graph.mul(self, node)

    def __truediv__(self, node):    # 重构除法
        return naive_Graph.div(self, node)

    def __sub__(self, node):    # 重构减法
        return naive_Graph.sub(self, node)

    def __pow__(self, node):
        return naive_Graph.pow(self, node)

    def __rpow__(self, node):  # 重构指数
        return naive_Graph.pow(node, self)

    """ 
    接下来还需要实现log, sin, cos的重载 
    但这些运算没有相应自带的重载运算符，所以以类方法的形式来实现。
    """
    def sin(self):
        return naive_Graph.sin(self)

    def cos(self):
        return naive_Graph.cos(self)

    def log(self):
        return naive_Graph.log(self)
```

重载后，当再发生x+y后，便会触发naive_Graph.add函数，返回被x和y指向的add节点。

至此，计算图的建图部分已经全部完成。



# Deriavative

我们针对八种算子定义了其前向传播，我们还需要定义这八种算子的梯度以进行反向传播。

对于一个Node指向一个Operator，我们需要求解从Operator流向Node的梯度，梯度的计算公式取决于Operator的种类（是add，还是mul，亦或是其他），计算需要带入的数值取决于Operator上游节点的值。

先定义一个查表函数，输入Operator和Node，根据Operator的类型来选择对应的求导函数，然后求导函数再根据Operator和Node进行计算：

```Python
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
```

新添其他算子后也可以在这里编写新算子的梯度。

然后定义每种算子的梯度：

```Python
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
```



# Forward pass

下一步就可以进行前向传播。计算图本质上是一个有向无环图（DAG），因此进行前向传播就是对这个DAG进行**拓扑排序**来得到一个线性序列。

这里有一个原则：若一个节点的入度为0，那么它的值要么是已知的，要么是当前可求的。

拓扑排序在这里的流程可以简述为：

- 构造一个队列Q。
- 找到所有入度为0的节点，将这些节点放入Q。（此时DAG里也有这些节点）
- 当Q内有元素时，重复执行一下步骤：
  - 从Q中删除一个节点n，并删除DAG中的节点n（删除DAG中的节点可以通过更改n的下游节点的入度来实现，见下面步骤）。
  - 遍历n的下游节点$n_i$
    - 修改$n_i$的入度。
    - 如果$n_i$的入度变成了0，意味着该节点可以求值了（因为该节点之前是下游节点，因此该节点一定是Operator类，可以通过Operator.calculate求值）。
    - 将$n_i$添加到队列Q首位。

运行完后所有Operator也就完成了前向传播。

```Python
@classmethod
def forward(cls):
    node_queue = []

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

```



# Backward propagation

和前向传播类似，反向传播实际上就是一个DAG的**逆拓扑排序**。

这里的一个原则是：所有出度为0的节点且不是Constant，当前梯度都可以传播到这里。

```Python
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
```

这里backward有两种模式：对所有输出节点求导 & 对某一输出节点求导。这两者在实现上非常容易，对所有输出节点求导则在一开始将所有出度为0的节点的梯度置为1，对某一输出节点求导则只将该节点梯度置1，其他出度为0的节点的梯度则为0.

然后将所有内容整合到**graph.py**里。



# Example

对函数$f(x)=log((x-7)^2+10)$找到最小值。

```Python
""" Example: Find the lowest value for function f """
x = Graph.Variable(6)  	# x初始值为6
f = ((x - 7) ** 2 + 10).log()
lr = 0.01  # random learning rate
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
```

优化过程如下：

![image](https://github.com/Charlie839242/mycs231n/blob/main/naive_imple_of_static_graph/img/Figure_1.png)  



对函数$f(x,y)=\frac{1}{2}x^2+xy+\frac{1}{2}y^2-2x-2y$找到最小值。

```Python
""" Example: Find the lowest value for a binary function  """
x, y = Graph.Variable(6), Graph.Variable(6)
f = 0.5 * x ** 2 + x * y + 0.5 * y ** 2 - 2 * x - 2 * y
lr = 0.01  # random learning rate
history = []
for _ in range(1000):
    Graph.forward()
    Graph.backward(f)
    x.value = x.value - x.grad * lr
    y.value = y.value - y.grad * lr
    Graph.kill_grad()
    history.append(f.value)
plt.plot(history)
plt.show()
Graph.clear()
```

优化过程如下：

![image](https://github.com/Charlie839242/mycs231n/blob/main/naive_imple_of_static_graph/img/Figure_2.png)  













