class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class Solution:
    def construct(self, nums):
        # [1, 2, 3, 4, 5]
        #        3
        #    2       4
        # 1              5
        root = self.helper(nums, 0, len(nums) - 1)
        return root

    def helper(self, nums, left, right):
        if left > right:
            return None
        middle = (left + right) // 2
        root = TreeNode(nums[middle])
        root.left = self.helper(nums, left, middle - 1)
        root.right = self.helper(nums, middle + 1, right)
        print('root: ', root.val)
        if root.left:
            print('root.left', root.left.val)
        if root.right:
            print('root.right', root.right.val)
        return root

    def preorder(self, root):
        # 3 2 4 1 5
        result = []
        self._preorder(root, result)
        return result

    def _preorder(self, root, result):
        if not root:
            return
        result.append(root.val)
        self._preorder(root.left, result)
        self._preorder(root.right, result)


if __name__ == '__main__':
    nums = [1, 2, 3, 4, 5, 6, 7]
    solution = Solution()
    root = solution.construct(nums)
    result = solution.preorder(root)
    print(result)