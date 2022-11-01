import random
from random import randint


class naive_Graph:
    """
    Define the class for the computational graph
    and the node in computational graph.
    """
    # 全局只有一个计算图，即node_list和id_list是全局唯一的
    # 且这样写可以不用实例化naive_Graph而操作和查看node_list和id_list
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


if __name__ == "__main__":
    random.seed(1)

    # Create two Nodes.
    node_1 = naive_Graph.Node()
    node_2 = naive_Graph.Node()
    naive_Graph.clear()

    

