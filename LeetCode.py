# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def printLstNode(self, lstNode):
        for i in range(len(lstNode)):
            print(lstNode[i].val)

    def printLinkedList(self, node):
        while node:
            print(node.val)
            node = node.next

    def createLinkedList(self, lstVal):
        head = ListNode(lstVal[0])
        tmpHead = head
        for i in range(1, len(lstVal)):
            tmpHead.next = ListNode(lstVal[i])
            tmpHead = tmpHead.next
        return head

    def addTwoNumbers(self, l1, l2):
        curL1 = l1
        curL2 = l2
        carry = 0
        totalNode = []
        count = 0
        while(curL1 or curL2):
            if (not curL1):
                total = curL2.val + carry
                curL2 = curL2.next
            elif (not curL2):
                total = curL1.val + carry
                curL1 = curL1.next
            else:
                total = curL1.val + curL2.val + carry
                curL1 = curL1.next
                curL2 = curL2.next

            carry = total // 10
            newNode = ListNode(total % 10)
            totalNode.append(newNode)
            if count != 0:
                totalNode[-2].next = totalNode[-1]
            count += 1

        if (carry != 0):
            newNode = ListNode(carry)
            totalNode[-1].next = newNode
        return totalNode[0]

    def lengthOfLongestSubstring(self, s):
        if len(s) == 1:
            return len(s)
        substring = []
        Max = 0
        for sub in s:
            print(sub)
            if sub in substring:
                if (len(substring) > Max):
                    Max = len(substring)
                if (substring.index(sub) != len(substring) - 1):
                    substring = substring[substring.index(sub) + 1:]
                else:
                    substring = []
                substring.append(sub)
            else:
                substring.append(sub)

        if len(substring) > Max:
            return len(substring)

        return Max

    def findMedianSortedArrays(self, nums1, nums2):
        if len(nums1) >= len(nums2):
            B = nums1
            A = nums2
        else:
            A = nums1
            B = nums2

        m = len(A)
        n = len(B)

        imin, imax = 0, m
        while(imin <= imax):
            i = (imin + imax) // 2
            j = (m + n - 2 * i) // 2
            if i > 0 and A[i - 1] > B[j]:
                imax = i - 1
            elif j > 0 and B[j - 1] > A[i]:
                imin = i + 1
            else:
                if i == 0:
                    max_left = B[j - 1]
                elif j==0:
                    max_left = A[i - 1]
                else:
                    max_left = max(A[i - 1], B[j - 1])

                if (m + n) % 2 == 1:
                   return max_left

                if i == m: min_right = B[j]
                elif j == n: min_right = A[i]
                else: min_right = min(A[i], B[j])

                return (max_left + min_right) / 2.0

    def convert(self, s, numRows):
        total = []
        for i in range(numRows):
            total.append([])

        pointer = 0
        reachBottom = False
        total[0].append(s[0])
        for c in s[1:]:
            if not reachBottom:
                pointer += 1
                total[pointer].append(c)
                if pointer == numRows - 1:
                    reachBottom = True
            else:
                pointer -= 1
                total[pointer].append(c)
                if pointer == 0:
                    reachBottom = False
        result = []
        for i in range(numRows):
            result = result + total[i]

        result = ''.join(result)
        return result

    def isMatch(self, s, p):
        if not p:
            return not s
        match = bool(s) and (p[0] == '.' or p[0] == s[0])

        if len(p) >= 2 and p[1] == '*':
            return self.isMatch(s, p[2:]) or match and self.isMatch(s[1:], p)
        else:
            return match and self.isMatch(s[1:], p[1:])

    def maxArea(self, height):
        maximum = 0
        start = 0
        end = len(height) - 1
        changeLeft = True
        while start != end:
            least = min((height[start], height[end]))
            area = least * (end - start)
            if area > maximum:
                maximum = area
            if changeLeft:
                start += 1
                changeLeft = False
            else:
                end -= 1
                changeLeft = True
        return maximum

    def letterCombinations(self, digits):
        # You can use dictionary
        lst = list('abcdefghijklmnopqrstuvwxyz')
        seperateLst = []
        index = 0
        for i in range(0, 8):
            if i == 5 or i == 7:
                num_ch = 4
            else:
                num_ch = 3
            tempLst = []
            for j in range(0, num_ch):
                tempLst.append(lst[index])
                index += 1
            seperateLst.append(tempLst)

        dialLst = []
        for c in digits:
            digit = int(c)
            dialLst.append(seperateLst[digit - 2])

        result = []
        index = 0
        def backtrack(combination, next_digit):
            if not next_digit:
                result.append(combination)
            else:
                for i in seperateLst[int(next_digit[0]) - 2]:
                    backtrack(combination + i, next_digit[1:])
        if digits:
            backtrack("", digits)
        return result

    def reverseKGroup(self, head, k):
        newhead = head
        index = 0
        ifReplaced = False
        while newhead != None:
            if (index+1) % k == 0:
                # do reverse
                lstNode = []
                tmpnext = newhead.next
                if not ifReplaced:
                    tmphead = head
                while (tmphead != newhead):
                    lstNode.append(tmphead)
                    tmphead = tmphead.next
                tmphead = tmphead.next
                tmptmphead = newhead
                for i in range(len(lstNode), 0, -1):
                    tmptmphead.next = lstNode[i - 1]
                    tmptmphead = tmptmphead.next
                tmptmphead.next = tmpnext
                if not ifReplaced:
                    ifReplaced = True
                    head = newhead
                print('tmphead val: ' + str(lstNode[-1].val))
            index += 1
            newhead = newhead.next

        return head

    def trap(self, height):
        if len(height) == 0:
            return 0
        left_max = [0] * len(height)
        rignt_max = [0] * len(height)
        area = 0
        left_max[0] = height[0]
        rignt_max[-1] = height[-1]
        for i in range(1, len(height)):
            left_max[i] = max(left_max[i - 1], height[i])

        for i in range(len(height) - 1, 1, -1):
            rignt_max[i - 1] = max(rignt_max[i], height[i - 1])

        for i in range(1, len(height)):
            area += min(left_max[i], rignt_max[i]) - height[i]

        return area

    def _dfs(self, grid, r, c):
        n_row = len(grid)
        n_col = len(grid[r])

        grid[r][c] = '0'
        if r - 1 >= 0 and grid[r - 1][c] == '1': self._dfs(grid, r - 1, c)
        if r + 1 < n_row and grid[r + 1][c] == '1': self._dfs(grid, r + 1, c)
        if c - 1 >= 0 and grid[r][c - 1] == '1': self._dfs(grid, r, c - 1)
        if c + 1 < n_col and grid[r][c + 1] == '1': self._dfs(grid, r, c + 1)

    def numIslands(self, grid):
        if len(grid) == 0:
            return 0
        n_islands = 0
        for i in range(len(grid[0])):
            for j in range(len(grid)):
                if grid[j][i] == '1':
                    self._dfs(grid, j, i)
                    n_islands += 1

        return n_islands

    def mergeTwoLists(self, l1, l2):
        prevHead = ListNode(-1)
        prev = prevHead

        while l1 and l2:
            if l1.val <= l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next

        prev.next = l1 if l1 is not None else l2
        return prevHead.next

    def num_islands(self, grid):
        def _dfs(self, grid, r, c):
            n_row = len(grid)
            n_col = len(grid[r])

            grid[r][c] = '0'
            if r + 1 <= n_row and grid[r + 1][c] == '1': self._dfs(grid, r + 1, c)
            if c + 1 <= n_col and grid[r][c + 1] == '1': self._dfs(grid, r, c + 1)

        if len(grid) == 0:
            return 0
        n_islands = 0
        for i in range(len(grid[0])):
            for j in range(len(grid)):
                if grid[j][i] == '1':
                    self._dfs(grid, j, i)
                    n_islands += 1
                print(grid)

        return n_islands

    def zigzag_level_order(self, root):
        if not root:
            return []

        answer = []
        self._helper(answer, root, 0, True)
        return answer

    def _helper(self, answer, root, depth, direction):
        if not root:
            return

        if len(answer) == depth:
            answer.append([])
        if not direction:
            answer[depth].append(root.val)
        if direction:
            answer[depth].insert(0, root.val)

        self._helper(answer, root.right, depth + 1, not direction)
        self._helper(answer, root.left, depth + 1, not direction)

    def combinationSum(self, candidates, target):
        def helper(self, target, candidates, idx, path, res):
            if target < 0 :
                return
            if target == 0:
                res.append(path)
                return
            for i in range(idx, len(candidates)):
                self.helper(target - candidates[i], candidates, i, path+[candidates[i]], res)

        res = []
        helper(target, candidates, 0, [], res)
        return res

    def kClosest(self, points, k):
        dist = lambda i: points[i][0]**2 + points[i][1]**2

        def sort(i, j, k):
            if i >= j: return

            K = (i + j) // 2
            points[i], points[K] = points[K], points[i]

            mid = partition(i, j)
            if k < mid - i + 1:
                sort(i, mid - 1, k)
            elif k > mid - i + 1:
                sort(mid + 1, j, k - (mid - i + 1))

        def partition(i, j):
            oi = i
            pivot = dist(i)
            i += 1

            while True:
                while i < j and dist(i) < pivot:
                    i += 1
                while i <= j and dist(j) >= pivot:
                    j -= 1
                if i >= j: break
                points[i], points[j] = points[j], points[i]

            points[oi], points[j] = points[j], points[oi]
            return j
        sort(0, len(points) - 1, k)
        return points[:k]

    def mostCommonWord(self, paragraph, banned):
        import re
        bag_words = re.findall(r'\w+', paragraph.lower())
        frequency = {}
        max_freq = 0
        answer = ""
        for word in bag_words:
            if word in banned: continue
            if not (word in frequency):
                frequency[word] = 1
            else:
                frequency[word] += 1
            if frequency[word] > max_freq:
                max_freq = frequency[word]
                answer = word
        return answer

    def isSubtree(self, mainTree, subTree):
        def equal(mainTree, subTree):
            if mainTree is None and subTree is None:
                return True
            if mainTree is None or subTree is None:
                return False
            return (mainTree.val == subTree.val and
                    equal(mainTree.left, subTree.left) and
                    equal(mainTree.right, subTree.right))
        def traverse(mainTree, subTree):
            return mainTree != None and (equal(mainTree, subTree) or
                                         traverse(mainTree.left, subTree) or
                                         traverse(mainTree.right, subTree))
        return traverse(mainTree, subTree)

    def partitionLabels(self, s):
        # get the last index of each char
        last = {c: i for i, c in enumerate(s)}
        ans = []
        anchor = j = 0
        for i, c in enumerate(s):
            j = max(j, last[c])
            if i == j:
                ans.append(i - anchor + 1)
                anchor = i + 1

        return ans

    def longestPalindrome(self, s):
        table = [[0 for i in range(len(s))]for j in range(len(s))]
        ans = ""
        for j in range(len(s) - 1, 0 - 1, -1):
            for i in range(j, len(s)):
                table[i][i] = True
                table[i][j] = s[j] == s[i] and (i - j < 3 or table[i - 1][j + 1])
                if table[i][j] and i - j + 1 > len(ans):
                    ans = s[j:i+1]

        return ans

    def prisonAfterNDays(self, cells, N):
        def getNextDay(cells):
            return [int(i > 0 and i < (len(cells) - 1) and cells[i-1] == cells[i+1])
                    for i in range(len(cells))]

        seen = {}
        while N > 0:
            c = tuple(cells)
            if c in seen:
                N = N % (seen[c] - N)
            seen[c] = N
            cells = getNextDay(cells)
            if N >= 1:
                N -= 1

        return cells

    def maxProfit(self, prices):
        min_price = max(prices)
        max_diff = 0
        min_index = len(prices) - 1
        for i, price in enumerate(prices):
            if price < min_price:
                min_price = price
                min_index = i
            if i > min_index and price - min_price > max_diff:
                max_diff = price - min_price

        return max_diff

    def hasPath(self, maze, start, destination):
        visited = [[False for i in range(len(maze[0]))]for j in range(len(maze))]
        def dfs(maze, start, destination, visited):
            if visited[start[0]][start[1]]:
                return False
            if start == destination:
                return True
            visited[start[0]][start[1]] = True
            r, l, u, d = start[1] + 1, start[1] - 1, start[0] - 1, start[0] + 1

            while(r < len(maze[0]) and maze[start[0]][r] == 0): # right
                r += 1
            if dfs(maze, [start[0], r - 1], destination, visited):
                return True

            while(l >= 0 and maze[start[0]][l] == 0):    # left
                l -=1
            if dfs(maze, [start[0], l + 1], destination, visited):
                return True

            while(d < len(maze) and maze[d][start[1]] == 0):
                d += 1
            if dfs(maze, [d - 1, start[1]], destination, visited):
                return True

            while(u >= 0 and maze[u][start[1]] == 0):
                u -= 1
            if dfs(maze, [u + 1, start[1]], destination, visited):
                return True


            return False

        return dfs(maze, start, destination, visited)

    def ladderLength(self, beginWord, endWord, wordList):
        from collections import defaultdict
        if endWord not in wordList or not beginWord or not endWord or not wordList:
            return 0

        all_combo_dict = defaultdict(list)
        L = len(beginWord)
        # Prepare a look a table for the word list
        for word in wordList:
            for i in range(L):
                all_combo_dict[word[:i] + '*' + word[i+1:]].append(word)

        queue = [(beginWord, 1)]
        visited = [beginWord]
        while queue:
            current_word, level = queue.pop(0)
            for i in range(L):
                intermediates = current_word[:i] + '*' + current_word[i+1:]
                for next_state in all_combo_dict[intermediates]:
                    if next_state == endWord:
                        return level + 1

                    if next_state not in visited:
                        queue.append((next_state, level + 1))
                        visited.append(next_state)
        return 0

    def diameterOfBinaryTree(self, root):
        self.length = 1
        def depth(root):
            if not root: return 0
            L = depth(root.left)
            R = depth(root.right)
            self.length = max(self.length, L + R + 1)
            return max(L, R) + 1

        depth(root)
        return self.length - 1

    def copyRandomList(self, head):
        if not head: return None

        # Insert a identity node without it's random pointer
        tmp = head
        while tmp:
            newNode = RandomListNode(tmp.label, None, None)
            newNode.next = tmp.next
            tmp.next = newNode
            tmp = tmp.next.next

        tmp = head
        # Move the random pointer to new Node
        while tmp:
            if tmp.random:
                tmp.next.random = tmp.random.next
            tmp = tmp.next.next

        # Seperate old list from new list
        newHead = head.next
        pold = head
        pnew = newHead
        while pnew.next:
            pold.next = pnew.next
            pold = pold.next
            pnew.next = pold.next
            pnew = pnew.next

        return newHead

    def floodFill(self, image, sr, sc, newColor):
        # DFS
        color = image[sr][sc]
        n_row = len(image)
        n_col = len(image[0])
        if image[sr][sc] == newColor: return image
        def dfs(r, c):
            if image[r][c] == color:
                image[r][c] = newColor
                if r > 0: dfs(r - 1, c)   # Moving up
                if r < n_row - 1: dfs(r + 1, c)   # Moving down
                if c > 0: dfs(r, c - 1)   # Moving left
                if c < n_col - 1: dfs(r, c + 1)   # Moving right
        dfs(sr, sc)
        return image

    def reverse(self, x):
        INT_MAX = 2 ** 31 - 1
        INT_MIN = -2 ** 31
        isNegative = x < 0
        if isNegative:
            x = x * -1
        rev = 0
        while x != 0:
            pop = x % 10
            x = x // 10
            if rev > INT_MAX // 10 or (rev == INT_MAX // 10 and pop > 7):
                return 0
            if -rev < INT_MIN // 10 or (-rev == INT_MIN // 10 and pop < -8):
                return 0

            rev = rev * 10 + pop

        if isNegative:
            return -rev
        return rev

    def myAtoi(self, sequence):
        INT_MAX = 2 ** 31 - 1
        INT_MIN = -2 ** 31
        switch = False
        isNegative = False
        whiteList = [str(ele) for ele in range(10)]
        answer = 0
        whiteList.append('-')
        whiteList.append('+')

        for ele in sequence:
            if ele == ' ' and switch:
                break
            if ele != ' ' and ele not in whiteList and not switch:
                return 0
            if ele != ' ' and ele not in whiteList and switch:
                break
            if ele != ' ' and ele in whiteList:
                if not switch:
                    whiteList.pop() # Took out '+'
                    whiteList.pop() # Took out '-'
                switch = True

                if ele == '-':
                    isNegative = True
                    continue
                if ele == '+':
                    continue
                answer = answer * 10 + int(ele)
        if isNegative:
            if -answer < INT_MIN:
                return INT_MIN
            return -answer
        if answer > INT_MAX:
            return INT_MAX
        return answer

    def isPalindrome(self, x):
        if x < 0:
            return False
        x = str(x)
        def helper(x):
            if not x:
                return True
            if x[0] != x[-1]:
                return False
            else:
                return helper(x[1:-1])
        return helper(x)

    def intToRoman(self, num):
        look_table = {1000: 'M',
                      500: 'D',
                      100: 'C',
                      50: 'L',
                      10: 'X',
                      5: 'V',
                      1: 'I',
                      }
        self.answer = ''
        def helper(num, base, divide_two, flag):
            if base == 0:
                return
            if base == 1000:
                for pop in range(num // base): self.answer = self.answer + look_table[base]
            else:
                if num // base == 4 and flag:   # Handle 9***
                    self.answer = self.answer[:-1] + look_table[base] + look_table[base * 10]
                elif num // base == 4 and (not flag):   # Handle 4***
                    self.answer = self.answer + look_table[base] + look_table[base * 5]
                else:
                    for _ in range(num // base):
                        self.answer = self.answer + look_table[base]
                if num > base:
                    flag = True
                else:
                    flag = False

            if divide_two:
                helper(num % base, base // 2, not divide_two, flag)
            else:
                helper(num % base, base // 5, not divide_two, flag)


        helper(num, 1000, True, False)
        return self.answer

    def romanToInt(self, sequence):
        look_table = {'M': 1000,
                      'D': 500,
                      'C': 100,
                      'L': 50,
                      'X': 10,
                      'V': 5,
                      "I": 1,
                      }
        self.answer = 0
        def helper(sequence):
            if len(sequence) == 0:
                return
            if sequence[0] == 'C':
                if len(sequence) > 1 and (sequence[1] == 'D' or sequence[1] == 'M'):
                    self.answer = self.answer + (look_table[sequence[1]] - look_table[sequence[0]])
                    helper(sequence[2:])
                else:
                    self.answer = self.answer + look_table[sequence[0]]
                    helper(sequence[1:])
            elif sequence[0] == 'X':
                if len(sequence) > 1 and (sequence[1] == 'L' or sequence[1] == 'C'):
                    self.answer = self.answer + look_table[sequence[1]] - look_table[sequence[0]]
                    helper(sequence[2:])
                else:
                    self.answer = self.answer + look_table[sequence[0]]
                    helper(sequence[1:])
            elif sequence[0] == 'I':
                if len(sequence) > 1 and (sequence[1] == 'V' or sequence[1] == 'X'):
                    self.answer = self.answer + look_table[sequence[1]] - look_table[sequence[0]]
                    print(self.answer)
                    helper(sequence[2:])
                else:
                    self.answer = self.answer + look_table[sequence[0]]
                    helper(sequence[1:])
            else:
                self.answer = self.answer + look_table[sequence[0]]
                helper(sequence[1:])

        helper(sequence)
        return self.answer

    def longestCommonPrefix(self, list_sequence):
        if len(list_sequence) == 0:
            return ""
        self.answer = list_sequence[0]
        def compare(s1, s2):
            idx = -1
            for i in range(min(len(s1), len(s2))):
                if s1[i] != s2[i]:
                    break
                idx = i
            return s1[:idx + 1]

        for i in range(1, len(list_sequence)):
            self.answer = compare(self.answer, list_sequence[i])

        return self.answer

    def removeDuplicates(self, nums):
        if len(nums) == 0: return 0
        slow_run = 0
        for fast_run in range(1, len(nums)):
            if nums[fast_run] != nums[slow_run]:
                slow_run += 1
                nums[slow_run] = nums[fast_run]
        nums = nums[:slow_run]
        return nums[:slow_run]

    def threeSum(self, nums):
        nums.sort()
        found = []

        for index, num in enumerate(nums):
            if num > 0:
                break
            if index > 0 and num == nums[index - 1]: continue
            left = index + 1
            right = len(nums) - 1
            while left < right:
                Sum = num + nums[left] + nums[right]
                if Sum == 0 and not [num, nums[left], nums[right]] in found:
                    found.append([num, nums[left], nums[right]])
                    right = right - 1
                elif Sum > 0:
                    right -= 1
                else:
                    left += 1

        return found

    def fourSum(self, nums, target):
        nums.sort()
        found = []
        for i in range(len(nums)):
            if target < 0 and nums[i] > 0:
                break
            if target > 0 and nums[i] > target:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            res = target - nums[i]
            for j in range(i + 1, len(nums)):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                left = j + 1
                right = len(nums) - 1
                while left < right:
                    L = [nums[i], nums[j], nums[left], nums[right]]
                    if sum(L) == target and L not in found:
                        found.append(L)
                        right = right - 1
                    elif sum(L) < target:
                        left += 1
                    else:
                        right -= 1

        return found

    def threeSumClosest(self, nums, target):
        nums.sort()
        dic = {}
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l = i + 1
            r = len(nums) - 1
            while l < r:
                sum = nums[i] + nums[l] + nums[r]
                if sum < target:
                    l += 1
                    dic[sum] = target - sum
                elif sum > target:
                    r -= 1
                    dic[sum] = sum - target
                else:
                    return target
        return min(dic, key=dic.get)


class MyQueue:
    def __init__(self):
        self.s1 = []

    def push(self, x):
        self.s1.append(x)
    def pop(self):
        if self.empty():
            return
        a = self.s1[0]
        if len(self.s1 == 1): self.s1 = []
        else: self.s1 = self.s1[1:]

        return a
    def peek(self):
        if self.empty(): return
        return self.s1[0]
    def empty(self):
        if len(self.s1) == 0:
            return True
        else: return False


class DLinkedNode():
    def __init__(self):
        self.key = 0
        self.value = 0
        self.prev = None
        self.next = None


class LRUCache:
    def _add_node(self, node):
        node.prev = self.head
        node.next = self.head.next

        self.head.next.prev = node
        self.head.next = node

    def _remove_node(selfs, node):
        prev = node.prev
        next = node.next

        prev.next = next
        next.prev = prev

    def _move_to_head(self, node):
        self._remove_node(node)
        self._add_node(node)

    def _remove_tail(self):
        last = self.tail.prev
        self._remove_node(last)
        return last

    def __init__(self, capacity):
        self.cache = {}
        self.capacity = capacity
        self.size = 0
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        node = self.cache.get(key)
        if not node:
            return -1
        self._move_to_head(node)
        return node.value

    def put(self, key, value):
        node = self.cache.get(key)  # See if the node exists
        if node:
            # node exists -> update the value and move to head
            node.value = value
            self._move_to_head(node)
        else:
            # node doesn't exist -> add new node
            node = DLinkedNode()
            node.key = key
            node.value = value

            self._add_node(node)
            self._move_to_head(node)
            self.size += 1

            self.cache[key] = node
            # Check if the size is more than the capacity
            if self.size > self.capacity:
                # Pop out tail
                tail = self._remove_tail()
                del self.cache[tail.key]
                self.size -= 1


class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None

import heapq
class KthLargest():
    def __init__(self, k, nums):
        self.k = k
        self.nums = nums
        heapq.heapify(self.nums)
        while len(self.nums) > k:
            heapq.heappop(nums)
    def add(self, val):
        if len(self.nums) < self.k:
            heapq.heappush(self.nums, val)
        else:
            heapq.heappushpop(self.nums, val)

        return self.nums[0]


class MinStack:

    def __init__(self):
        self.stack = [(None, float('inf'))]
    def push(self, x):
        self.stack.append((x, min(x, self.stack[-1][1])))
    def pop(self):
        if len(self.stack) > 1: self.stack.pop()
    def top(self):
        return self.stack[-1][0]
    def getMin(self):
        return self.stack[-1][-1]
