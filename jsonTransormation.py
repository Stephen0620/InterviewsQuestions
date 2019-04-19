class File():
    def __init__(self, ID, name, children):
        self.ID = ID
        self.name = name
        self.children = children
        self.children_list = []
    
class Solution():
    def __init__(self, list_children):
        self.list_children = list_children
        self.look_table = {}
        for i, ele in enumerate(list_children):
            self.look_table[ele.ID] = ele
            
    def show_input(self):
        for ele in self.list_children:
            string = "id: {ID}, name: {name}, children: {children}".format(
                    ID = ele.ID, name = ele.name, children = ele.children)
            print(string)
    def transform(self):
        self._dfs(self.list_children[0])
        
    def _dfs(self, root):
        if len(root.children) == 0:
            return
        for i, ele in enumerate(root.children):
            # Mistake during the interview, 
            # I was writing root.children_list.append(ele)
            # This is appending the IDs not the node
            
            root.children_list.append(self.look_table[ele])
            self._dfs(self.look_table[ele])
        
        return root
    
    def show_info(self):
        def helper(root, n_indent):
            string = ''
            for i in range(n_indent):
                string += '\t'
            string += "id: {ID}, name: {name}, children: {children}".format(
                    ID = root.ID, name = root.name, children = root.children)
            print(string)
            for ele in root.children_list:
                helper(ele, n_indent + 1)
        helper(self.list_children[0], 0)
# Test case
if __name__ == '__main__':
    # root level             root
    # level 0        child1        child2
    # level 1    child3       child4    child 5
    # Output shoulde be like:
    # info of root:
    #   info for child1
    #       info for child3
    #   info for child2
    #       info for child4
    #       info for child5
    root = File(12345, "root", [2345, 4567])
    child1 = File(2345, "child1", [888])
    child2 = File(4567, "child2", [999, 111])
    child3 = File(888, "child3", [])
    child4 = File(999, "child4", [])
    child5 = File(111, "child5", [])
    
    solution = Solution([root, child1, child2, child3, child4, child5])
    print("Input: ")
    solution.show_input()
    solution.transform()
    print("\nOutput: ")
    solution.show_info()
    
    