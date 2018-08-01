# -- coding: utf-8 --
import dataStru as ds


def duplicate(inputs):
    """
        数组中重复的数字
    """

    if inputs == None or len(inputs) <= 0: return False;

    n = len(inputs)

    for i in range(0, n):

        while(i != inputs[i]):
            tmp = inputs[i]

            if inputs[i] == inputs[inputs[i]]:
                duplicate_num = inputs[i]
                break
            inputs[i], inputs[tmp]= inputs[tmp], inputs[i]

    return duplicate_num      


def find(inputs, target): 
    """
        二维数组中的查找
    """
    
    if inputs == None or len(inputs) <= 0: return False;
    M = len(inputs) #row
    N = len(inputs[0]) #col

    i = M - 1
    j = 0
    
    while(i >= 0 and j < N):
        if target == inputs[j][i]:
            return True
        elif target < inputs[j][i]:
            i -= 1
        elif target > inputs[j][i]:
            j += 1
    return False 


def replaceSpace(inputs):
    """
        替换空格
    """

    if inputs == None or len(inputs) <= 0: return False;

    oldlen = len(inputs)
    for i in range(oldlen):
        if inputs[i] == ' ':
            inputs += '  '

    p1 = oldlen - 1
    p2 = len(inputs) - 1

    inputs = list(inputs)

    while(p1 >= 0 and p2 > p1):
        if(inputs[p1] == ' '):
            inputs[p2] = '0'
            p2 -= 1
            inputs[p2] = '2'
            p2 -= 1
            inputs[p2] = '%'
            p2 -= 1
        else:
            inputs[p2] = inputs[p1]
            p2 -= 1
        p1 -= 1

    s=''.join(inputs)
    return  s


def printListFromTailToHead1(inputs):
    """
    从尾到头打印链表, 使用栈
    """
    print("Use stack...")
    stack = ds.stack()
    while(inputs != None):
        stack.push(inputs.data)
        inputs = inputs.next
    
    rev = []
    while(stack.top != None):
        rev.append(stack.pop())

    return rev

def printListFromTailToHead2(inputs):
    """
    从尾到头打印链表，使用递归
    """
    print("Use recursion...")

    if inputs != None:
        printListFromTailToHead2(inputs.next)
        print(inputs.data)

def printListFromTailToHead3(inputs):
    """
    从尾到头打印链表，使用头插法,头结点和第一个节点的区别：头结点是在头插法中使用的一个额外节点，这个节点不存储值；第一个节点就是链表的第一个真正存储值的节点。
    """
    print("Use head...")
    head = ds.Node(-1)
    
    while(inputs != None):
        new_node = inputs.next
        inputs.next = head.next
        head.next = inputs
        inputs = new_node

    rev = []
    node = head.next
    while(node != None):
        rev.append(node.data)
        node = node.next
    
    return rev
    
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
def reConstructBinaryTree(pre, tin):
    """
    根据二叉树的前序遍历和中序遍历的结果，重建出该二叉树。
    """   
    # write code here
    if len(pre) == 0:
        return None
    elif len(pre) == 1:
        return TreeNode(pre[0])
    else:
        
        root = TreeNode(pre[0])
        root.left = reConstructBinaryTree(pre[1:tin.index(pre[0])+1], tin[: tin.index(pre[0])])
        root.right = reConstructBinaryTree(pre[tin.index(pre[0])+1:], tin[tin.index(pre[0]) + 1:])

    return root

class Solution:
    """
    用两个栈实现队列
    """
    def __init__(self):
        self.stack_a = []
        self.stack_b = []
        
    def push(self, node):
        # write code here
        self.stack_a.append(node)
        
    def pop(self):
        if self.stack_b:
            return self.stack_b.pop()
        else:
            while(self.stack_a):
                self.stack_b.append(self.stack_a.pop())

            return self.stack_b.pop()

def minNumberInRotateArray(rotateArray):
    """
    旋转数组的最小数字,根据旋转数组前部分大于后部分。设置了两个指针，一个指向数组头head，另一个指向数组尾部end。当head > end, 数组尾部指针左移，一直到head < end, 说明数组顺序大小突变。

    """
    if len(rotateArray) == 0: return 0
        
    head = 0
    end = len(rotateArray) - 1

    if head == end: return rotateArray[0]
    else:
        while(head < end):
            if rotateArray[head] >= rotateArray[end]:
                end -= 1
            else:
                min_num = rotateArray[end + 1]
                break
        return min_num


def Fibonacci(n):
    # write code here
    if n == 0: return 0
    elif n <= 2: return 1
    elif n <= 39:
        # return Fibonacci(n - 1) + Fibonacci(n - 2)
        num = [1] * (n + 1)  
        for i in range(3, n + 1):
            num[i] =  num[i - 1] + num[i - 2]
        return num[n]


def factorial(m):
    if m == 0: return 1
    else:
        mm = m
        for i in range(1, m):
            mm = mm * (m - i)
        return mm

def jumpFloor(number):
    # write code here
    """
    一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
   
    画图算的呃Orz... 列了方程式：x + 2y = n; x + y = k; (限定条件x <= n and y <= n/2); x + y = k这条直线在x + 2y = n以及限定条件内滑动，可以知道在满足条件的情况下k有多个值，每种值排列组合一下是C(k, y) ，根据方程式算出k = n - y。最后累计计算C(n - y, y)就行了。
    太笨啦，想了半天...
    """
    y = 0
    count = 0 
    if number == 1: return 1
    else:
        while(y <= number/2):
            m = factorial(number - y)
            n = factorial(y)
            m_n = factorial(number - 2*y)

            count += m/(n*m_n)
            y += 1

        return count

def jumpFloorII(number):
    # write code here
    """
    ## 题目描述

        一只青蛙一次可以跳上 1 级台阶，也可以跳上 2 级……它也可以跳上 n 级。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
    """
    if number == 0: 
        return 0
    elif number == 1: 
        return 1
    elif number == 2: 
        return 2
    else:
        num = [1] * (number + 1)  
        num[2] = 2
        for i in range(3, number + 1):           
            s = 0
            for j in range(1, i):
                s += num[i - j]
                
            num[i] = s + 1
        
        return num[number]


def rectcover(n):
    """
    我们可以用 2\*1 的小矩形横着或者竖着去覆盖更大的矩形。请问用 n 个 2\*1 的小矩形无重叠地覆盖一个 2\*n 的大矩形，总共有多少种方法？
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    elif n == 2:
        return 2
    elif n >= 3:
        # return rectcover(n - 1) + rectcover(n - 2)
        num = [0, 1, 2] + [1]*(n - 2) 
        for i in range(3, n + 1):
            num[i] =  num[i - 1] + num[i - 2]
        return num[n]


def NumberOf1(n):
    """
    ???输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
    """
    count = 0
    if n < 0:
        n = n & 0xffffffff
    while n:
        count += 1
        n = (n - 1) & n
    return count

def FindKthToTail(head, k):
    """
        输入一个链表，输出该链表中倒数第k个结点。
    """
    # write code here
    if head == None or k <=0:
        return None
    slow = head.head
    fast = head.head
    while k > 1:
        if fast.next != None:
            fast = fast.next
            k -= 1
        else:
            return None
    
    while fast.next != None:
        fast = fast.next
        slow = slow.next
        
    return slow.data

def reOrderArray(array):
    """
    输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，
    所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
    """


class Solution:
    def __init__(self):
        self.stack = []
    def push(self, node):
        # write code here
        return self.stack.append(node)
    def pop(self):
        # write code here
        self.stack.pop()
        return self.stack
    def top(self):
        # write code here
        return self.stack[-1]
    def min(self):
        # write code here
        if len(self.stack)==1 :
            return self.stack[0]
        while len(self.stack)>1:
            m1 = self.top()
            self.pop()
            m2 = self.top()
            if (m1 < m2):
                m = m1
            else:
                m = m2
        
        return m

def IsPopOrder(pushV, popV):
    # write code here 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。

    stack = []
    
    while popV:
        if stack and stack[-1] == pushV[-1]:
            stack.pop()
            pushV.pop()
        else:
            stack.append(popV[0])
            popV.pop(0)
    
    
    if len(stack) == 1 and stack[-1] == pushV[-1]:
        return True
    else:
        return False



        
def FindPath(root, expectNumber):
    # write code here
    #输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
    if root == None:
        return []
    
    if root and root.left == None and root.right == None and root.data == expectNumber:
        return [[root.data]]
    
    left = FindPath(root.left, expectNumber - root.data)
    right = FindPath(root.right, expectNumber - root.data)
    
    res = []
    for i in left + right:
        res.append([root.data] + i)
    
    return res

def Permutation(s):
    # write code here
    #输入一个字符串,按字典序打印出该字符串中字符的所有排列。
    if len(s) <= 1:
        return [s]
    sl = []
    for i in range(len(s)):
        for j in Permutation(s[0:i] + s[i + 1:]):
            sl.append(s[i] + j)
    
    
    return sl


def movingCount(threshold, rows, cols):
    # write code here
    board = [[0 for _ in range(cols)] for _ in range(rows)]
    def block(r,c):
        s = sum(map(int,str(r)+str(c)))
        return s>threshold
    class Context:
        acc = 0
    def traverse(r,c):
        if not (0<=r<rows and 0<=c<cols): return
        if board[r][c]!=0: return
        if board[r][c]==-1 or block(r,c):
            board[r][c]=-1
            return
        board[r][c] = 1
        Context.acc+=1
        traverse(r+1,c)
        traverse(r-1,c)
        traverse(r,c+1)
        traverse(r,c-1)
    traverse(0,0)
    return Context.acc

def FindGreatestSum(array):
    #给出一个长度为n的序列A1,A2,…,An,求最大连续和。换句话说，要求找到1<=i <= j<=n，使得Ai+Ai+1+..+Aj尽量大
    # max_sum = array[0]
    # temp = 0
    # for i in array:
    #     if temp < 0:
    #         temp = i
    #     else:
    #         temp += i
    #     max_sum = max(max_sum, temp)
    
    # return max_sum
    dp = array
    
    for i in range(len(array)):
        if i == 0:
            dp[i] = array[i]
        else:
            tmp = dp[i - 1] + array[i]
            dp[i] = max(tmp, array[i])
            print dp[i]
            
    return max(dp)

def GetNumberOfK(data, k):
    data = map(str, data)
    s = 0
    for i in data:
        for j in i:
            if j == str(k):
                s += 1
    return s


#层次遍历
def levelOrder(r):
    #输入一颗二叉树，求树的深度
    if r is None:
        return 0
    count = 0
    q = [] #模拟一个队列储存结点
    q.append(r)
    while len(q) != 0:
        tmp = [] #使用列表储存同层结点
        for i in range(len(q)):
            t = q.pop(0)
            if t.left is not None:
                q.append(t.left)
            if t.right is not None:
                q.append(t.right)
            tmp.append(t.data)
        if tmp:
            count += 1
  
    return count


def FindFirstCommonNode(head1, head2):
    #两个链表的第一个公共结点
    if head1 is None or head2 is None:
        return None

    L1 = []
    L2 = []
    while head1:
        L1.append(head1.data)
        head1 = head1.next
    while head2:
        L2.append(head2.data)
        head2 = head2.next
    
    f = None
    while L1 and L2:
        top1 = L1.pop(-1)
        top2 = L2.pop(-1)
        if top1 == top2:
            f = top1
        else:
            break
    
    return f


if __name__ == "__main__":
    
    L1 = ds.ListNode()
    for i in [1, 3, 8, 5, 2]:
        L1.add(i)        
    # L1.print_ListNode()

    L2 = ds.ListNode()
    for i in [8, 5, 2]:
        L2.add(i)        
    # L2.print_ListNode()

    print(FindFirstCommonNode(L1.head, L2.head))

    
    # print GetNumberOfK([12,3,13], 1)

    # print(FindGreatestSum([6,-3,-2,7,-15,1,2,2]))
    

    # print(movingCount(5, 10, 10))

    # perm_nums = Permutation('abc')
    # no_repeat_nums = list(set(perm_nums))
    # print('perm_nums', len(perm_nums), perm_nums)
    # print('no_repeat_nums', len(no_repeat_nums), no_repeat_nums)

    # tree1 = ds.Tree(data=8)
    # tree2 = ds.Tree(data=9)
    # tree3 = ds.Tree(tree1,data=6)
    # tree4 = ds.Tree(tree2,None,data=7)
    # base = ds.Tree(tree3,tree4,5)
    
    # print(FindPath(base, 19))
    



    # #1
    # print("Duplicate number is:", duplicate([2,5,1,0,3,5]))

    # #2
    # matrix = [
    # [1,   4,  7, 11, 15],
    # [2,   5,  8, 12, 19],
    # [3,   6,  9, 16, 22],
    # [10, 13, 14, 17, 24],
    # [18, 21, 23, 26, 30]
    # ]
    # print(find(matrix, target = input("Input a integer number:")))

    # #3
    # print(replaceSpace('We Are Happy'))

    # #4
    # L = ds.ListNode()
    # for i in [1, 3, 8, 5, 2]:
    #     L.add(i)        
    # L.print_ListNode()
    # print(printListFromTailToHead1(L.head))
    # print(printListFromTailToHead2(L.head))
    # print(printListFromTailToHead3(L.head))

    # #5
    # preorder = [3,9,20,15,7]
    # inorder =  [9,3,15,20,7]
    # print(reConstructBinaryTree(preorder, inorder))

    # #6
    # s = Solution()
    # s.push(2)
    # s.push(3)
    # print(s.pop(), s.pop())

    # #7
    # print(minNumberInRotateArray([2,2,3,1,1]))

    # #8
    # print(Fibonacci(2))

    #9
    # print(jumpFloor(2))

    #10
    # print(jumpFloorII(5))

    #11
    # print(rectcover(5))

    #12
    # print(NumberOf1(-10))

    #13
    # L = ds.ListNode()
    # for i in [1, 3, 8, 5, 2]:
    #     L.add(i)        
    # L.print_ListNode()
    # print(FindKthToTail(L, 3))

    # s = Solution()
    # s.push(1)
    # print(s.min())
    # s.push(2)
    # print(s.min())

    # print(IsPopOrder([1,2,3,4,5],[3,5,4,1,2]))