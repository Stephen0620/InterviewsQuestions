class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        index = {}            
        for i, x in enumerate(nums):
            check = target - x
            if check in index:
                return [index[check], i]
            else:
                index[x] = i

solution = Solution()
index = solution.twoSum([3,2,4], 6) 
print(index)           