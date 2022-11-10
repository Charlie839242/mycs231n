from typing import List, Tuple, Union
import numpy as np


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

    """首先是Tensor和ndarray相似的属性，像shape, ndim等等"""
    # @property将类方法转化为一个属性，调用时是Tensor.is_leaf而非Tensor.is_leaf()
    @property
    def is_leaf(self):
        """判断是否为叶节点: requires_grad=False,没有上游节点，二者满足一个就是叶节点"""
        return (not self.requires_grad) or len(self.last) == 0

    @property
    def shape(self) -> Tuple[int]:
        """张量的形状，用法同NumPy."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """张量的维度，用法同NumPy."""
        return self.data.ndim

    @property
    def dtype(self):
        """张量的数据类型，用法同NumPy."""
        return self.data.dtype

    @property
    def size(self) -> int:
        """张量的元素个数，用法同NumPy."""
        return self.data.size

    """
    然后是对Tensor的一些和ndarray的操作，像reshape，max，astype等等
    像astype这种不可导的操作，可以直接对self.data用numpy的API来操作
    像reshape这种可导的操作，应该被看作算子，要写出其前向传播和反向传播函数
    和之前的naive动态图中一样，先写出一/二元函数模板，再通过模板来定义这些算子
    """
    def astype(self, new_type):
        """类型转换，我们不允许可求导节点的类型转换"""
        assert not self.requires_grad
        self.data.astype(new_type)

    def reshape(self, *new_shape):
        """张量的reshape，用法同NumPy."""
        return reshape(self, new_shape)

    @property
    def T(self):
        """张量的转置，用法同NumPy."""
        return self.transpose()

    def transpose(self, *axes):
        return transpose(self, axes if len(axes) != 0 else None)

    def max(self, axis: Union[int, Tuple, None] = None, keepdims=False):
        """找到张量中的最大值，用法同NumPy."""
        return max(self, axis, keepdims)

    def min(self, axis: Union[int, Tuple, None] = None, keepdims=False):
        """找到张量中的最小值，用法同NumPy."""
        return min(self, axis, keepdims)

    def mean(self, axis: Union[int, Tuple, None] = None, keepdims=False):
        """找到张量的平均值，用法同NumPy."""
        return mean(self, axis, keepdims)

    def sum(self, axis: Union[int, Tuple, None] = None, keepdims=False):
        """找到张量中的和，用法同NumPy."""
        return sum(self, axis, keepdims)

    def argmax(self, axis: Union[int, Tuple, None] = None):
        """找到张量中最大值的索引，用法同NumPy."""
        return argmax(self, axis)

    def argmin(self, axis: Union[int, Tuple, None] = None):
        """找到张量中最小值的索引，用法同NumPy."""
        return argmin(self, axis)

    def build_edge(self, tensor):
        """构建两节点的有向边，正常不适用"""
        self.next.append(tensor)
        tensor.last.append(self)

    def __repr__(self) -> str:
        return "{0}({1}, requires_grad={2})".format("Tensor", self.data, self.requires_grad)

    """重载运算符"""
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

    def __truediv__(self, x):
        """overload division: /"""
        return div(self, x)

    def __rtruediv__(self, x):
        return div(x, self)

    def __pow__(self, x):
        """overload power operation: **"""
        return pow(self, x)

    def __rpow__(self, x):
        return pow(x, self)

    def __pos__(self):
        """
        overload unary positive operation: +
        different from addition
        This is a unary operation
        """
        return 1 * self

    def __neg__(self):
        """overload unary negative operation: -"""
        return -1 * self

    def __abs__(self):
        """overload unary absolute operation: abs()"""
        return abs(self)

    def __getitem__(self, key):
        """
        overload indexing(索引) operation: self[key]
        Follow the rules of indexing in NumPy

        Parameters
        ----------
        key: Union[int, condition, ndarrray, Tensor, slice]

        Examples
        ----------
        self[0], self[self<=0], self[np.array([0,1,2])], self[0:2]
        """
        return get_slice(self, key)

    def __setitem__(self, key, value):
        """
        overload indexing(索引)/assignment(赋值) operation: self[key] = value
        不允许self允许求导，因为该操作不可导，且该操作是in-place operation

        Parameters
        ----------
        key : Union[int, condition, ndarrray, Tensor, slice]
        value : Union[number, ndarrray, Tensor]

        Example
        -------
        x = Tensor([1, 2, 3])
        x[x <= 2] = 0
        x
        <[0 0 3], int64, Tensor>
        """
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(key, Tensor):
            key = key.data
        self.data[key] = value.data if isinstance(value, Tensor) else value

    def __len__(self):
        """overload len function: len(self)"""
        return len(self.data)

    """
    In-place Operation: 原地操作符
    指改变一个tensor的值的时候，不经过复制操作，而是直接在原来的内存上改变它的值。
    有两种情况下是不能使用原地操作符的：
        - 对于requires_grad=True的叶子张量不能使用 inplace operation
          - 这一点非常严格，因为inplace operation可能把一个叶子张量变成非叶子张量。
          -  Example：
             >>> a = torch.tensor([10., 5., 2., 3.], requires_grad=True)
             >>> print(a, a.is_leaf)
             >>> # tensor([10.,  5.,  2.,  3.], requires_grad=True) True
             
             >>> a[:] = 0
             >>> print(a, a.is_leaf)
             >>> # tensor([0., 0., 0., 0.], grad_fn=<CopySlices>) False
             我们看到，在进行对 a 的重新 inplace 赋值之后，表示了 a 是通过 
             copy slice operation 生成的，grad_fn 都有了，所以自然而然不是叶子节点了
        - 对于在求梯度阶段需要用到的张量不能使用 inplace operation
          - 例如y = 2*x，对x反向传播的梯度应该是2，但如果在前向传播之后修改2，计算结果会出现错误。
    可以想象，满足requires_grad=False且不参与反向传播的叶子张量，一定是几个requires_grad=False
    的张量计算得到一个新的requires_grad=False的张量。这种情况下可以用inplace operation。
    """

    def __iadd__(self, other):
        """
        overload compound addition: self+=other
        该操作不可导，所以需要requires_grad=False
        """
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data += other
        return self

    def __isub__(self, other):
        """
        overload compound subtraction: self-=other
        该操作不可导，所以需要requires_grad=False
        """
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data -= other
        return self

    def __imul__(self, other):
        """
        overload compound multiplication: self*=other
        该操作不可导，所以需要requires_grad=False
        """
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data *= other
        return self

    def __itruediv__(self, other):
        """
        overload compound division: self/=other
        该操作不可导，所以需要requires_grad=False
        """
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data /= other
        return self

    def __imatmul__(self, other):
        """
        overload compound matrix multiplication: self@=other
        该操作不可导，所以需要requires_grad=False
        """
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data @= other
        return self

    """重载比较运算符, 比较操作不需要记录梯度"""
    def __lt__(self, other):
        """overload < function"""
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data < other)

    def __le__(self, other):
        """overload <= function"""
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data <= other)

    def __eq__(self, other):
        """overload == function"""
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data == other)

    def __ne__(self, other):
        """overload != function"""
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data != other)

    def __gt__(self, other):
        """overload > function"""
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data > other)

    def __ge__(self, other):
        """overload >= function"""
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data >= other)

    def retain_grad(self):
        """This function works only if it is a non-leaf tensor so that its grad will be retained"""
        self.retain_grad = True

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
        for i in range(len(Graph.node_list) - 1, -1, -1):   # from len(Graph.node_list)-1 to 0
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

    def zero_grad(self):
        """梯度归零"""
        self.grad = np.zeros(self.shape)

    def numpy(self) -> np.ndarray:
        """返回Tensor的内部数据，即NumPy数组(深拷贝)"""
        return self.data.copy()

    def item(self, *args):
        """返回Tensor.data指定索引处的"""
        return self.data.item(args)


"""
Operator也具有data、grad等属性
但它同时还记录了前向反向传播等信息
所以Operator类继承自Tensor类
"""


class UnaryOperator(Tensor):
    """
    一元运算算子的基类:
    每个具体的一元算子，像log()等等都继承该类
    并重写forward和grad_fn方法
    """
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
    """
    二元运算算子的基类
    每个具体的二元算子，像add等等都继承该类
    并重写forward和grad_fn方法
    """
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


class add(BinaryOperator):
    """加法算子"""
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data + y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        return grad


class sub(BinaryOperator):
    """减法算子"""
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


class pow(BinaryOperator):
    """幂运算算子"""
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data**y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray) -> np.ndarray:
        if node is self.last[0]:
            return (self.data * self.last[1].data / node.data) * grad
        else:
            return self.data * self.xp.log(self.last[0].data) * grad


class matmul(BinaryOperator):
    """矩阵乘法算子"""
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return x.data @ y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray) -> np.ndarray:
        """
        Consider the case that ndim = 2:
        z = x @ y
        dx = upstream_grad @ y.T
        dy = x.T @ upstream-grad
        """
        if node is self.last[0]:
            if self.last[1].ndim == 1:
                """
                z = x @ y, y.shape=(n,), 
                dx = upstream_grad @ y.T
                前向传播中y是被升维到n×1后再计算的
                但计算结果的shape最后一个1又是被去掉了的
                因此为了正常反向传播:
                要把self的梯度的最后一个维度加上,
                以及y的最后一个维度加上
                详见numpy的broadcast机制
                """
                return np.expand_dims(grad, -1) @ np.expand_dims(self.last[1].data, 0)
            elif self.last[1].ndim > 2:
                """
                z = x @ y
                dx = upstream_grad @ y.T
                当y.ndim > 2, 转置y的最后两个axis即可
                注意当x.ndim < y.ndim时求得的x的梯度和x的形状不同
                这是由于NumPy的broadcast机制造成的
                在Tensor.backward里面已经处理了这种情况
                """
                shape = list(range(self.last[1].ndim))
                shape[-1], shape[-2] = shape[-2], shape[-1]
                return grad @ self.last[1].data.transpose(*shape)
            else:
                """
                z = x @ y
                dx = upstream_grad @ y.T
                """
                return grad @ self.last[1].data.T
        else:
            if self.last[0].ndim == 1:
                return np.expand_dims(self.last[0].data, -1) @ np.expand_dims(grad, -2)
            elif self.last[0].ndim > 2:
                shape = list(range(self.last[0].ndim))
                shape[-1], shape[-2] = shape[-2], shape[-1]
                return self.last[0].data.transpose(*shape) @ grad
            return self.last[0].data.T @ grad


class abs(UnaryOperator):
    """取绝对值算子"""
    def __init__(self, x: Tensor):
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.abs(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        mask = np.zeros(x.shape)
        mask[x > 0] = 1.
        mask[x < 0] = -1.
        return grad * mask


class sum(UnaryOperator):
    """求和算子"""
    def __init__(self, x: Tensor, axis=None, keepdims=False):
        """
        :param x: the sum of Tensor x is desired
        :param axis: summing operation performed on which axis
        :param keepdims: whether to keep x.ndim unchanged
        """
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.sum(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        """
        self.keepdims=False时，x在指定axis处的维度会直接消失
        例如x.shape = (2, 3, 4), y = x.sum(axis=0)
        则有y.shape = (3, 4)
        当多个这样的维度消失后，会影响broadcast
        所以需要将这些维度以1的形式加回来: np.expand_dims
        一种特殊情况是axis=None时，np.sum得到的是一个标量
        self.grad也是一个标量，此时可以正常传播，不需要expand_dims
        """
        if not (self.axis is None or self.keepdims):
            grad = np.expand_dims(grad, axis=self.axis)
        return np.ones(x.shape) * grad


class mean(UnaryOperator):
    """求均值算子"""
    def __init__(self, x: Tensor, axis=None, keepdims=False):
        """
        :param x: the mean of Tensor x is desired
        :param axis: mean operation performed on which axis
        :param keepdims: whether to keep x.ndim unchanged
        """
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.mean(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if not (self.axis is None or self.keepdims):
            grad = np.expand_dims(grad, axis=self.axis)
        # self.data.size < x.data.size
        return np.ones(x.shape) * grad * self.data.size / x.data.size


class max(UnaryOperator):
    """求最大值算子"""
    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        """
        :param x: the maximum value of Tensor x is desired
        :param axis: max operation performed on which axis
        :param keepdims: whether to keep x.ndim unchanged
        """
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.max(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.keepdims or self.axis is None:
            """这种情况下self.data == x.data 可以通过broadcast比较"""
            full_dim_y = self.data
        else:
            """这种情况下self.data == x.data 不可以通过broadcast比较，需要还原维度"""
            full_dim_y = np.expand_dims(self.data, axis=self.axis)
            grad = np.expand_dims(grad, axis=self.axis)
        # 注意: (full_dim_y == x.data).dtype = bool
        return (full_dim_y == x.data).astype(float) * grad


class min(UnaryOperator):
    """求最小值算子"""
    def __init__(self, x: Tensor, axis=None, keepdims=False):
        """
        :param x: the minimum value of Tensor x is desired
        :param axis: min operation performed on which axis
        :param keepdims: whether to keep x.ndim unchanged
        """
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.min(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            full_dim_y = np.expand_dims(self.data, axis=self.axis)
            grad = np.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(float) * grad

# 由于argmax、argmin等算子不可导
# 所以不可能出现在计算图中，也不会有grad_fn记录反向传播
# 所以这些算子可以直接继承Tensor


class argmax(Tensor):
    """求最大值索引算子: 不可导"""
    def __init__(self, x: Tensor, axis=None):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.axis = axis
        super().__init__(self.forward(x))

    def forward(self, x: Tensor) -> np.ndarray:
        return np.argmax(x.data, axis=self.axis)


class argmin(Tensor):
    """求最小值索引算子: 不可导"""
    def __init__(self, x: Tensor, axis=None):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.axis = axis
        super().__init__(self.forward(x))

    def forward(self, x: Tensor) -> np.ndarray:
        return np.argmin(x.data, axis=self.axis)


class exp(UnaryOperator):
    """指数运算算子"""
    def __init__(self, x: Tensor):
        super().__init__(x)

    def forward(self, x: Tensor):
        return np.exp(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return self.data * grad


class log(UnaryOperator):
    """对数运算算子"""
    def forward(self, x: Tensor):
        return np.log(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad / x.data


class maximum(BinaryOperator):
    """逐一比较x、y每个位置取最大值"""
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return np.maximum(x.data, y.data)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x.data) * grad


class minimum(BinaryOperator):
    """逐一比较x、y每个位置取最小值"""
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return np.minimum(x, y)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x) * grad


def sqrt(x: Tensor):
    """平方根函数"""
    return x**0.5


def square(x: Tensor):
    """平方函数"""
    return x * x


class reshape(UnaryOperator):
    """reshape算子"""
    def __init__(self, x: Tensor, new_shape: tuple):
        self.new_shape = new_shape
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data.reshape(self.new_shape)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(x.shape)


class transpose(UnaryOperator):
    """transpose算子"""
    def __init__(self, x: Tensor, axes: tuple = None) -> None:
        self.axes = axes
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.transpose(self.axes)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.axes is None:
            return grad.transpose()
        # np.argsort(self.axes) 和 self.axes是一对反操作
        return grad.transpose(tuple(np.argsort(self.axes)))


class get_slice(UnaryOperator):
    """切片算子"""
    def __init__(self, x: Tensor, key) -> None:
        if isinstance(key, Tensor):
            self.key = key.data
        else:
            self.key = key
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data[self.key]

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        full_grad = np.zeros(x.shape)
        full_grad[self.key] = grad
        return full_grad


class concatenate(Tensor):
    """张量拼接算子"""
    def __init__(self, tensors: List[Tensor], axis=0):
        """
        :param tensors: a group of tensors that need concat
        :param axis: concat operation performed on which axis

        Example:
        x.shape = (5, 4)    y.shape = (3, 4)
        (np.concatenate([x, y], axis=0)).shape = (8, 4)
        """
        requires_grad = False
        self.tensors = tensors
        self.axis = axis
        self.indices = [0]

        for i in range(len(self.tensors)):
            assert isinstance(tensors[i], Tensor), "Concatenate elements in 'tensors' must be 'Tensor'"

            requires_grad = requires_grad or self.tensors[i].requires_grad
            self.indices.append(self.indices[-1] + self.tensors[i].shape[self.axis])
            """
            Example: 
            x.shape=(3, 4), y.shape=(4, 4), z.shape=(5, 4)
            w = np.concatenate([x, y, z], axis=0)
            Therefore: self.indices = [0, 3, 7, 12]
            w[self.indices[0], self.indices[1]] 就是 x
            w[self.indices[1], self.indices[2]] 就是 y
            w[self.indices[2], self.indices[3]] 就是 z
            """
        super().__init__(self.forward(), requires_grad=requires_grad)
        if self.requires_grad:
            for i in range(len(self.tensors)):
                self.tensors[i].build_edge(self)

    def forward(self):
        return np.concatenate([t.data for t in self.tensors], axis=self.axis)

    def grad_fn(self, x: Tensor, grad: np.ndarray):
        """截取grad中的一部分"""
        x_id = self.tensors.index(x)
        start = self.indices[x_id]
        end = self.indices[x_id + 1]
        slc = [slice(None)] * grad.ndim
        slc[self.axis] = slice(start, end)
        return grad[tuple(slc)]


# 一些包装的特殊矩阵
def zeros(shape, dtype=None, requires_grad=False):
    """similar to np.zeros"""
    return Tensor(np.zeros(shape), dtype=dtype, requires_grad=requires_grad)


def ones(shape, dtype=None, requires_grad=False):
    """similar to np.ones"""
    return Tensor(np.ones(shape), dtype=dtype, requires_grad=requires_grad)


def randn(*shape, dtype=None, requires_grad=False):
    """similar to np.random.randn"""
    return Tensor(np.random.randn(*shape), dtype=dtype, requires_grad=requires_grad)


def rand(*shape, dtype=None, requires_grad=False):
    """similar to np.random.rand"""
    return Tensor(np.random.rand(*shape), dtype=dtype, requires_grad=requires_grad)


def uniform(low: float, high: float, shape=None, dtype=None, requires_grad=False):
    """similar to np.random.uniform"""
    return Tensor(np.random.uniform(low, high, size=shape), dtype=dtype, requires_grad=requires_grad)


def empty(shape, dtype=None, requires_grad=False):
    """similar to np.random.empty"""
    return Tensor(np.empty(shape), dtype=dtype, requires_grad=requires_grad)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
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





