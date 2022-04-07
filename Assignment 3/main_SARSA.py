import numpy as np
import random
import csv


class Node:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.value = 0
        self.up = 0
        self.down = 0
        self.left = 0
        self.right = 0

    def get_print_action(self):
        if self.value!= 0:
            return self.value
        else:
            max_value = max([self.up, self.down, self.left, self.right])

def random_grid(n, m):
    grid = [[Node(row, col) for col in range(m)] for row in range(n)]

    # for i in range(n):
    #     for j in range(m):
    #         grid[i][j] = Node((i, j))

    a = n*m/10
    p_terminal = random.randint(1, int(a))
    n_terminal = random.randint(1, int(a))
    barrier = random.randint(1, int(a))

    i = 0
    while i < p_terminal:
        row = random.randint(0, n-1)
        col = random.randint(0, m-1)
        if grid[row][col].value == 0:
            grid[row][col].value = random.randint(1, 9)
            i += 1

    i = 0
    while i < n_terminal:
        row = random.randint(0, n-1)
        col = random.randint(0, m-1)
        if grid[row][col].value == 0:
            grid[row][col].value = random.randint(-9, -1)
            i += 1

    i = 0
    while i < barrier:
        row = random.randint(0, n-1)
        col = random.randint(0, m-1)
        if grid[row][col].value == 0:
            grid[row][col].value = 'X'
            i += 1

    start_decision = random.choice([0, 1])

    if start_decision == 1:
        row = random.choice([0, n-1])
        col = random.randint(0, m-1)

    else:
        row = random.randint(0, n-1)
        col = random.choice([0, m-1])

    grid[row][col].value = 'S'

    return grid, [row, col]

def determineAction(s):
    epsilon = 0.6
    if random.random() < epsilon:
        action = random.choice(['up', 'down', 'left', 'right'])
    else:
        max_value = max(s.up, s.down, s.left, s.right)
        choice = []
        if s.up == max_value:
            choice.append('up')
        if s.down == max_value:
            choice.append('down')
        if s.left == max_value:
            choice.append('left')
        if s.right == max_value:
            choice.append('right')

        action = random.choice(choice)

    return action

def simple_takeAction(s, a, grid):
    if a == 'up':
        if s.row == 0:
            s_new = [s.row, s.col]
        else:
            if grid[s.row - 1][s.col].value == 'X' or grid[s.row - 1][s.col].value == 'S':
                s_new = [s.row, s.col]
            else:
                s_new = [s.row - 1, s.col]

    elif a == 'down':
        if s.row == len(grid) - 1:
            s_new = [s.row, s.col]
        else:
            if grid[s.row + 1][s.col].value == 'X' or grid[s.row + 1][s.col].value == 'S':
                s_new = [s.row, s.col]
            else:
                s_new = [s.row + 1, s.col]

    elif a == 'left':
        if s.col == 0:
            s_new = [s.row, s.col]
        else:
            if grid[s.row][s.col - 1].value == 'X' or grid[s.row][s.col - 1].value == 'S':
                s_new = [s.row, s.col]
            else:
                s_new = [s.row, s.col - 1]

    elif a == 'right':
        if s.col == len(grid[0]) - 1:
            s_new = [s.row, s.col]
        else:
            if grid[s.row][s.col + 1].value == 'X' or grid[s.row][s.col + 1].value == 'S':
                s_new = [s.row, s.col]
            else:
                s_new = [s.row, s.col + 1]

    return s_new

def update_SARSA(grid, s, a, s_new):
    gamma = 0.2
    alpha = 0.1
    R = -0.04

    if s_new.value != 0:
        if a == 'up':
            Q_s_new = random.choice([s_new.up, s_new.left, s_new.right])
            grid[s.row][s.col].up = s.up + alpha*(R + gamma*Q_s_new - s.up)

        elif a == 'down':
            Q_s_new = random.choice([s_new.down, s_new.left, s_new.right])
            grid[s.row][s.col].down = s.down + alpha*(R + gamma*Q_s_new - s.down)

        elif a == 'left':
            Q_s_new = random.choice([s_new.up, s_new.down, s_new.left])
            grid[s.row][s.col].left = s.left + alpha*(R + gamma*Q_s_new - s.left)

        elif a == 'right':
            Q_s_new = random.choice([s_new.up, s_new.down, s_new.right])
            grid[s.row][s.col].right = s.right + alpha*(R + gamma*Q_s_new - s.right)

    else:
        if a == 'up':
            Q_s_new = s_new.value
            grid[s.row][s.col].up = s.up + alpha*(R + gamma*Q_s_new - s.up)

        elif a == 'down':
            Q_s_new = s_new.value
            grid[s.row][s.col].down = s.down + alpha*(R + gamma*Q_s_new - s.down)

        elif a == 'left':
            Q_s_new = s_new.value
            grid[s.row][s.col].left = s.left + alpha*(R + gamma*Q_s_new - s.left)

        elif a == 'right':
            Q_s_new = s_new.value
            grid[s.row][s.col].right = s.right + alpha*(R + gamma*Q_s_new - s.right)


    return grid

def rl(grid, start):
    n = 0
    while n < 1000:
        print(n)
        n = n+1
        s = grid[start[0]][start[1]]
        while s.value == 0 or s.value == 'S':
            a = determineAction(s)
            # print(a)
            s_new_pos = simple_takeAction(s, a, grid)
            s_new = grid[s_new_pos[0]][s_new_pos[1]]
            grid = update_SARSA(grid, s, a, s_new)
            s = s_new
            # print(s.value)

    print(s.value)

    return grid



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    grid, start = random_grid(5, 3)    #coordinate in form (row, col)
    print(len(grid[0]))
    print(start)
    solved_grid = rl(grid, start)

    # for row in grid:
    #     for col in row:




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
