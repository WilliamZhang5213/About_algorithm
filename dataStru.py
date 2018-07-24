
#linked list
class Node(object):  
    """
        node initialization

        A node consists of data and next pointer
    """
    def __init__(self, data):
        self.data = data
        self.next = None

class ListNode(object):
    def __init__(self):
        self.head = None

    def length(self):
        length = 0
        p = self.head
        while p:
            length += 1
            p = p.next
        return length

    def add(self, data):
        node = Node(data)
        
        if self.head is None:
            self.head = node
        else:
            p = self.head
            while p.next:
                p = p.next
            p.next = node

    def print_ListNode(self):
        if self.length() == 0:
            print("Empty ListNode")
        else:
            node = self.head
            while node:
                print(node.data)
                node = node.next


#stack
class stack(object):
    def __init__(self):
        self.top = None
    
    def isTop(self):
        if self.top == None:
            return None
        else:
            return self.top.data

    def push(self, item):
        node = Node(item)
        node.next = self.top
        self.top = node

    def pop(self):
        if self.isTop() == None:
            return None
        else:
            pdata = self.top.data
            self.top = self.top.next
            return pdata


    def print_stack(self):
        if self.isTop() == None:
            print("Empty Stack")
        else:
            while(self.isTop()):
                print(self.pop())

# if __name__ == '__main__':
#     L = ListNode()
#     for i in [1, 3, 8]:
#         L.add(i)
        
#     L.print_ListNode()

    # s = stack()
    # s.push(1)
    # s.push('23')
    # s.print_stack()