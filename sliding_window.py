class Solution:
    def get_unique(self, arr, window_size):
        self.prev_run = {}
        ans = []
        for i in range(len(arr) - window_size + 1):
            if i == 0:
                for ele in arr[i:i + window_size]:
                    if ele not in self.prev_run:
                        self.prev_run[ele] = 1
                    else:
                        self.prev_run[ele] += 1
            else:
                if arr[i:i + window_size][-1] not in self.prev_run:
                    self.prev_run[arr[i:i + window_size][-1]] = 1
                else:
                    self.prev_run[arr[i:i + window_size][-1]] += 1
            ans.append(len(self.prev_run))
            self.prev_run[arr[i:i + window_size][0]] -= 1
            if self.prev_run[arr[i:i + window_size][0]] == 0:
                del self.prev_run[arr[i:i + window_size][0]]
                
        return ans

if __name__ == '__main__':
    solution = Solution()
    result = solution.get_unique([1, 1, 3, 4, 2, 2, 1, 4, 5], 3)
    print('test array: ', [1, 1, 3, 4, 2, 2, 1, 4, 5], ', window_size: ', 3)
    print('answer: ', result)