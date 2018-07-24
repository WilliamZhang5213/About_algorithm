# -- coding: utf-8 --

class BTreeNode:
    def __init__(self, data, left = None, right = None):
        self.data = data
        self.left = left
        self.right = right
    
    
def num_nodes(root):
    #求二叉树的结点个数
    num = 0
    if not root:
        return 0
    else:
        left = num_nodes(root.left)
        right = num_nodes(root.right)
        return 1 + left + right

#先根遍历递归
def preorder(t):
    if not t:
        print("^", end=' ')
        return 
    print("(" + str(t.data), end=' ')
    preorder(t.left)
    preorder(t.right)
    print(")", end=' ')

#非递归
def preorder_norec(t):
    s = [] 
    while t is not None or s != []:
        while t is not None:
            print(t.data, end=' ')
            s.append(t.right)
            t = t.left
        t = s.pop(-1)

#中根遍历递归
def midorder(t):
    if not t:
        print("^", end=' ')
        return 
    print("(", end=' ')
    preorder(t.left)
    print(str(t.data), end=' ')
    preorder(t.right)
    print(")", end=' ')

#中根非递归遍历
def midorder_norec(t):
    s = [] 
    while t is not None or s != []:
        while t is not None:
            s.append(t)
            t = t.left
        
        if s != []:
            t = s.pop(-1)
            print(t.data, end=' ')
            t = t.right


if __name__ == "__main__":
    t = BTreeNode(1, BTreeNode(2), BTreeNode(3))
    print(preorder(t))
    print(preorder_norec(t))

    print(midorder(t))
    print(midorder_norec(t))