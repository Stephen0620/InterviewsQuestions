def get_next(matrix, cur_node, direction, visited):
    direction_table = {
        (0, 1): 0,
        (1, 0): 1,
        (0, -1): 2,
        (-1, 0): 3,
    }
    if direction == 0:
        # Means we're walking toward the right side, so we check the right side first.
        diff = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    if direction == 1:
        # Means we're walking toward the down side, so we check the down side first.
        diff = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    if direction == 2:
        # Means we're walking toward the left side, so we check the left side first.
        diff = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    if direction == 3:
        # Means we're walking toward the up side, so we check the up side first.
        diff = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    for idx, dir in enumerate(diff):
        next_point = (cur_node[0] + dir[0], cur_node[1] + dir[1])
        # return if the next point is valid and also return the direction of the next point.
        if is_valid(matrix, next_point, visited):
            return next_point, direction_table[dir]
    return None, None


def is_valid(matrix, point, visited):
    if point[0] < 0 or point[0] >= len(matrix):
        return False
    if point[1] < 0 or point[1] >= len(matrix[0]):
        return False
    if point in visited:
        return False
    return True


def spiral_matrix(matrix):
    if not matrix:
        return []
    from collections import deque
    queue = deque([(0, 0)])
    direction = 0   # 0: right, 1: down, 2: left, 3 up.
    visited = set()
    result = []

    # BFS
    while queue:
        # cur_node is the index of current position.
        cur_node = queue.popleft()
        visited.add(cur_node)
        result.append(matrix[cur_node[0]][cur_node[1]])
        next_node, direction = get_next(matrix, cur_node, direction, visited)
        # If there is no next node, means we there is no un-visited node. So we already go the point in the matrix
        if not next_node:
            return result
        queue.append(next_node)

    return result


def print_matrix(matrix):
    for row in range(len(matrix)):
        row_info = []
        for col in range(len(matrix[row])):
            row_info.append(str(matrix[row][col]))
        msg = ', '.join(row_info)
        print(msg)


if __name__ == '__main__':
    matrix = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [1, 2, 3]]
    print_matrix(matrix)
    print('Expected: ', '1, 2, 3, 6, 9, 3, 2, 1, 7, 4, 5, 8')
    print('Result:', spiral_matrix(matrix))