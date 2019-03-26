# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        
class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        
        count = 0
        total1 = 0
        while True:
            total1 += l1.val * (10**count)
            if l1.next == None:
                break
            else:
                count += 1
                l1 = l1.next
                
        return total1
    
l1 = ListNode(2)
l1.next = ListNode(4)
l1.next.next = ListNode(3)

l2 = ListNode(5)
l2.next = ListNode(6)
l2.next.next = ListNode(4)

solution = Solution()
solution.addTwoNumbers(l1, l2)