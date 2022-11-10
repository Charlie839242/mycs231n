之前我们实现了针对scalar的动态图机制，现在我们来基于NumPy提供的张量运算接口来实现实现针对多维张量的动态图。



### Graph

像之前一样，定义一个全局的类Graph：

```Python
class Graph:
    node_list = []

    @classmethod
    def add_node(cls, node):
        """添加静态图节点"""
        cls.node_list.append(node)

    @classmethod
    def clear(cls):
        """清空计算图"""
        cls.node_list.clear()

    @classmethod
    def free_graph(cls):
        """释放计算图(保留叶节点)，一般用于反向传播后"""
        new_list = []
        for node in Graph.node_list:
            node.next.clear()
            if node.is_leaf:
                new_list.append(node)
            # 因为is_leaf要通过上游节点数量判断
            # 所以判断后再清空上游节点
            node.last.clear()
        Graph.node_list = new_list

    @classmethod
    def zero_grad(cls):
        """删除计算图所有梯度信息，一般用于更新权重后"""
        for node in Graph.node_list:
            node.grad = 0 if node.requires_grad else None
```

Graph需要具有清空计算图、释放计算图和清空梯度信息的功能



### Tensor

将`ndarray`封装成具有梯度等信息的`Tensor`。

```Python
class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False, retain_grad=False):
        """
        :param data: ndarray / Tensor / anything can be converted by np.array
        :param dtype: dtype that you want self.data to be
        :param requires_grad: decide whether calculate gradient
        :param retain_grad: works only if it is a non-leaf tensor so that the grad of the non-leaf tensor is retained
        """
        # 若传入的data是一个tensor
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype)

        self.retain_grad = retain_grad
        self.requires_grad = requires_grad
        assert not (self.requires_grad and self.dtype != float), \
            "Only Tensors of floating point dtype can require gradients!"
        self.grad = np.zeros_like(self.data) if self.requires_grad else None

        # 上下游节点: list[Tensor]
        self.next = list()
        self.last = list()

        if self.requires_grad:
            # 不需要求梯度的节点不出现在动态计算图中
            Graph.add_node(self)
```

`Tensor.data`就是`numpy.ndarray`，其他的attributes封装了梯度等其他信息。

`numpy.ndarray`的attributes，像`shape`、`ndim`、`dtype`属性，我们希望`Tensor`也具有这些属性，即通过Tensor.shape也能得到`Tensor.data.shape`一样的结果，因此我们对`Tensor`进一步封装一些被`@property`修饰的method（注意这里是获取`ndarray`的属性而不涉及到修改`ndarray`本身，因此调用这些不涉及到反向传播的梯度传播，不用单独定义类）：

```Python
@property
def shape(self) -> Tuple[int]:
    '''张量的形状，用法同NumPy.'''
    return self.data.shape


@property
def ndim(self) -> int:
    '''张量的维度，用法同NumPy.'''
    return self.data.ndim


@property
def dtype(self):
    '''张量的数据类型，用法同NumPy.'''
    return self.data.dtype


@property
def size(self) -> int:
    '''张量的元素个数，用法同NumPy.'''
    return self.data.size
```



类似的，`numpy.ndarray`的methods，像`ndarray.transpose(`), ndarray.reshape()等等一些方法，我们希望`Tensor`也具有这样的一些方法，即通过`Tensor.reshape`能得到和`Tensor.data.reshape()`一样的结果，因此我们对Tensor进一步封装一些method（注意像`reshape`这种操作是属于operator/算子一类的，也就是说这种操作在计算图里面是要计算梯度的，因此对这些操作我们在后面单独定义它的类，并在类中单独写出其前向传播和反向传播）：

```Python
def astype(self, new_type):
    '''类型转换，我们不允许可求导节点的类型转换'''
    assert not self.requires_grad
    self.data.astype(new_type)


def reshape(self, *new_shape):
    '''张量的reshape，用法同NumPy.'''
    return reshape(self, new_shape)


@property
def T(self):
    '''张量的转置，用法同NumPy.'''
    return self.transpose()


def transpose(self, *axes):
    return transpose(self, axes if len(axes) != 0 else None)


def max(self, axis: Union[int, Tuple, None] = None, keepdims=False):
    '''找到张量中的最大值，用法同NumPy.'''
    return max(self, axis, keepdims)


def min(self, axis: Union[int, Tuple, None] = None, keepdims=False):
    '''找到张量中的最小值，用法同NumPy.'''
    return min(self, axis, keepdims)


def mean(self, axis: Union[int, Tuple, None] = None, keepdims=False):
    '''找到张量的平均值，用法同NumPy.'''
    return mean(self, axis, keepdims)


def sum(self, axis: Union[int, Tuple, None] = None, keepdims=False):
    '''找到张量中的和，用法同NumPy.'''
    return sum(self, axis, keepdims)


def argmax(self, axis: Union[int, Tuple, None] = None):
    '''找到张量中最大值的索引，用法同NumPy.'''
    return argmax(self, axis)


def argmin(self, axis: Union[int, Tuple, None] = None):
    '''找到张量中最小值的索引，用法同NumPy.'''
    return argmin(self, axis)
```

像`ndarray.astype`这种method显然是不可导的，因此这种method只能对`requires_grad=False`的Tensor使用。

像`ndarray.reshape`这种method是可导的，需要考虑计算它的梯度，因此我们在后面单独定义了`reshape`类，其中包含了前向传播公式和反向传播公式。



### Backward Propagation

同样，我们将反向传播定义为一个计算图中一个节点的类方法，也就是从特定节点来传播梯度。这里有几个新参数：

- `retain_graph`: 动态图正常来说前向传播一次就销毁了。若是要多次反向传播，则可以令`retain_graph=True`。
- `retain_grad`: 这个是梯度传播沿途的节点的属性。反向传播后，为了节约内存，所有非叶节点的梯度都会被置None，但若是想要知道某个非叶节点的梯度，则令其`retain_grad=True`。

```Python
def backward(self, retain_graph=False):
    """
    backward from a node
    retain_graph: 是否保留计算图
    """
    if self not in Graph.node_list:
        print("AD failed because the node is not in graph.")
        return

    assert self.data.ndim == 0, "backward should be called only on a scalar."

    self.grad = np.ones_like(self.data)
    for i in range(len(Graph.node_list) - 1, -1, -1):  # from len(Graph.node_list)-1 to 0
        # 找到self在node_list中的索引
        if Graph.node_list[i] is self:
            y_id = i
            break

    for node in Graph.node_list[y_id::-1]:  # [y_id, y_id-1, y_id-2···, 0]
        grad = node.grad
        for last in [l for l in node.last if l.requires_grad]:
            add_grad = node.grad_fn(last, grad)
            """
            广播机制处理梯度
            add_grad和last.grad的shape有可能不同，但这二者的shape一定是可以传播的。
            但传播有可能将last.grad的shape变成add_grad的shape，这就直接改变了last-grad的值。
            所以要对add_grad按照broadcast规则进行处理，使得add_grad的shape等于last_grad的shape
            """
            if add_grad.shape != last.shape:
                add_grad = np.sum(
                    add_grad,
                    axis=tuple(-i for i in range(1, last.ndim + 1)
                               if last.shape[-i] == 1),
                    keepdims=True,
                )
                add_grad = np.sum(
                    add_grad,
                    axis=tuple(range(add_grad.ndim - last.ndim)),
                )
            last.grad += add_grad

        # 当一个node的所有上游节点计算完成后
        # 若该node是非叶节点且retain_grad=False，则删除grad信息
        if (not node.is_leaf) and (not self.retain_grad):
            node.grad = None

    if not retain_graph:
        Graph.free_graph()
```



### Calculations between Tensor: Operator

类似的，我们通过定义一元函数类模板和二元函数类模板来实现Tensor之间的运算。

因为Operator也具有data、grad属性，但同时还需要记录前向反向传播的公式，所以Operator类继承自Tensor类。

```Python
class UnaryOperator(Tensor):
    '''
    一元运算算子的基类:
    每个具体的一元算子，像log()等等都继承该类
    并重写forward和grad_fn方法
    '''
    def __init__(self, x: Tensor):
        """
        :param x: self的上游节点
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        super().__init__(data=self.forward(x), requires_grad=x.requires_grad)
        if self.requires_grad:
            x.build_edge(self)

    def forward(self, x: Tensor) -> np.ndarray:
        """
        前向传播函数
        :param x: self的上游节点
        :return: ndarray，前向传播的结果，应该赋值给self.data
        """
        raise NotImplementedError

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        """
        反向传播函数
        :param x: Tensor，self的上游节点
        :param grad: ndarray，流入self的梯度
        :return: ndarray，流入x的梯度
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return "Tensor({0}, op={1})".format(self.data, self.__class__.__name__)
    
    
class BinaryOperator(Tensor):
    '''
    二元运算算子的基类
    每个具体的二元算子，像add等等都继承该类
    并重写forward和grad_fn方法
    '''
    def __init__(self, x: Tensor, y: Tensor):
        """
        :param x: self的上游节点
        :param y: self的上游节点
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if not isinstance(y, Tensor):
            y = Tensor(y)
        super().__init__(data=self.forward(x, y), requires_grad=x.requires_grad or y.requires_grad)
        if self.requires_grad:
            x.build_edge(self)
            y.build_edge(self)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        """
        前向传播函数
        :param x: self的上游节点之一
        :param y: self的上游节点之一
        :return: ndarray，前向传播的结果
        """
        raise NotImplementedError

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        """
        反向传播函数
        :param x: self的上游节点之一
        :param grad: 流入self的梯度
        :return: 流入x的梯度
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return "Tensor({}, op={})".format(self.data, self.__class__.__name__)
```

forward和grad_fn方法都将在具体的算子中重写。以加减乘除的Operator为例：

```Python
class add(BinaryOperator):
    '''加法算子'''
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y)
        
    def forward(self, x: Tensor, y: Tensor):
        return x.data + y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        return grad


class sub(BinaryOperator):
    '''减法算子'''
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y)
        
    def forward(self, x: Tensor, y: Tensor):
        return x.data - y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        if node is self.last[0]:
            return grad
        return -grad

    
class mul(BinaryOperator):
    """逐元素乘法算子"""
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data * y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        if node is self.last[0]:
            return self.last[1].data * grad
        return self.last[0].data * grad


class div(BinaryOperator):
    """除法算子"""
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data / y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        temp = grad / self.last[1].data
        if node is self.last[0]:
            return temp
        return -self.data * temp
```



### Overload Operators

接下来重载Tensor的运算符，以使得能够自动前向传播。

部分运算符如下：

```Python
def __add__(self, x):
    """overload addition: +"""
    return add(self, x)


def __radd__(self, x):
    return add(x, self)


def __sub__(self, x):
    """overload subtraction: -"""
    return sub(self, x)


def __rsub__(self, x):
    return sub(x, self)


def __mul__(self, x):
    """overload normal multiplication: +"""
    return mul(self, x)


def __rmul__(self, x):
    return mul(x, self)


def __matmul__(self, x):
    """overload matrix multiplication: @"""
    return matmul(self, x)


def __rmatmul__(self, x):
    return matmul(x, self)
```

至此一个支持高维张量的动态图系统就构建好了。



### Example

我们考虑一个全连接层：

```Python
y = x @ w + b
------------------------------
"""
x.shape = (50, 3072)
w.shape = (3072, 10)
b.shape = (10, )
"""
```

x是固定的，我们希望找到合适的w和b以使得y中每个元素都接近于0。理想情况也就是说：

```Python
print((y ** 2).sum == 0)
-----------------------------
True
```

因此可以构建计算图，其中x， w， b都随机生成：

```Python
np.random.seed(1)
lr = 1e-5
x = rand(50, 3072, requires_grad=False)
w = rand(3072, 10, requires_grad=True)
b = rand(10, requires_grad=True)
history = []
for i in range(1000):
    y = (x @ w + b) ** 2
    loss = y.sum()
    history.append(loss.item())
    loss.backward()
    w.data = w.data - lr * w.grad
    b.data = b.data - lr * b.grad
    Graph.zero_grad()
plt.plot(history[300:1000])
plt.show()
```

最终优化过程如下：

![image](https://github.com/Charlie839242/mycs231n/blob/main/vectorized_imple_of_dynamic_graph/img/Figure_2.png)  

