import numpy as np
import random
import csv
import copy
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class CSV_:

    def __init__(self, path):
        self.CSVfilePath = path

    def creatCSV_board(self, board, cost, file_address='E:\Study\Sem 2\Artificial Intelligence\Project3\\Gridworld.csv'):
        # default file address
        if len(file_address) == 0:
            file_address = self.CSVfilePath
        with open(file_address, 'a', newline='') as CSVfile:
            write = csv.writer(CSVfile)
            write.writerows(board)
            write.writerow([cost])
        CSVfile.close()

    def readCSV_board(self, file_address='E:\Study\Sem 2\Artificial Intelligence\Project3\\Gridworld.csv'):

        # default file address
        if len(file_address) == 0:
            file_address = self.CSVfilePath

        board = []

        with open(file_address, mode='r', encoding='utf-8-sig')as file:
            csvFile = csv.reader(file)

            for lines in csvFile:
                board.append(list(lines))

            for rows in range(len(board)):
                print(board[rows])

        file.close()
        return board

class Node:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.terminal = 0
        self.tag = None

        # Q values for all 4 possible actions
        self.Q_up = 0
        self.Q_down = 0
        self.Q_right = 0
        self.Q_left = 0

        # keep track of number of time agent took the same action
        self.counter_up = 0
        self.counter_down = 0
        self.counter_right = 0
        self.counter_left = 0
        self.counter = 0
        self.iter = 0

    def get_Q_value(self, action):
        if action == "up":
            return self.Q_up
        elif action == "down":
            return self.Q_down
        elif action == "right":
            return self.Q_right
        elif action == "left":
            return self.Q_left
        else:
            print("Error: WRONG ACTION GIVEN")

    def update(self, action, value):
        if action == "up":
            self.Q_up = value
        elif action == "down":
            self.Q_down = value
        elif action == "right":
            self.Q_right = value
        elif action == "left":
            self.Q_left = value
        else:
            print("Error: WRONG ACTION GIVEN")

    def get_max_action(self):
        actions = [self.Q_up, self.Q_down, self.Q_right, self.Q_left]
        action = max(actions)
        index = actions.index(action)
        if index == 0:
            return "up"
        if index == 1:
            return "down"
        if index == 2:
            return "right"
        if index == 3:
            return "left"

    def get_print_action(self):
        if self.tag is None or self.tag == "START":
            max_value = max([self.Q_up, self.Q_down, self.Q_left, self.Q_right])
            # print(max_value)
            if max_value <= 0:
                x = 0
                y = 0
            elif max_value == self.Q_up:
                x = -0.1
                y = 0
            elif max_value == self.Q_down:
                x = 0.1
                y = 0
            elif max_value == self.Q_left:
                x = 0
                y = -0.1
            elif max_value == self.Q_right:
                x = 0
                y = 0.1
            else:
                x = 0
                y = 0

            if self.tag == "START":
                state = self.tag
            else:
                state = 0

            # print(x, y)

        elif self.tag == "BARRIER":

            x = 0
            y = 0
            state = self.tag
        else:
            x = 0
            y = 0
            state = self.terminal
        return x, y, state

    def avg_timeOnNode(self, total_iter):
        # counter = self.counter_up + self.counter_down + self.counter_left + self.counter_right
        # print("counter = ", self.counter)
        # print("total iter = ", total_iter)
        # print("__________________")

        return int(self.counter*100/total_iter)

    def counter_update(self, action):
        if action == "up":
            self.counter_up += 1
        elif action == "down":
            self.counter_down += 1
        elif action == "right":
            self.counter_right += 1
        elif action == "left":
            self.counter_left += 1
        else:
            print("Error: WRONG ACTION GIVEN")

    def total_counter_update(self, iter):
        if self.iter != iter:
            self.counter += 1

        self.iter = iter

    def get_count(self, action):
        if action == "up":
            return self.counter_up
        elif action == "down":
            return self.counter_down
        elif action == "right":
            return self.counter_right
        elif action == "left":
            return self.counter_left
        else:
            print("Error: WRONG ACTION GIVEN")

class world:

    def __init__(self, file, reward_per_action, gamma, time_to_learn, prob_of_moving):
        self.file = file
        self.reward_per_action = reward_per_action
        self.gamma = gamma
        self.time_to_learn = time_to_learn
        self.prob_of_moving = prob_of_moving
        self.alpha = 0.2
        self.min_count = 5
        self.barrier = []
        self.start = []

    def get_grid(self):
        csv_ = CSV_(self.file)
        self.grid = csv_.readCSV_board()
        self.node_World = [[Node(i, j) for j in range(len(self.grid[0]))] for i in range(len(self.grid))]

        for i, ele in enumerate(self.grid):
            for j, e in enumerate(ele):
                if "X" == e:
                    self.barrier.append([i, j])
                    self.node_World[i][j].tag = "BARRIER"
                elif "S" == e:
                    self.start =[i,j]
                    self.node_World[i][j].tag = "START"
                elif abs(int(e)) > 0:
                    self.node_World[i][j].terminal = int(e)
                    self.node_World[i][j].tag = "Terminal"

        print("BARRIER: ", self.barrier)
        print("START: ", self.start)
        print(self.grid)

    def random_grid(self, n, m):
        grid = [[Node(row, col) for col in range(m)] for row in range(n)]
        self.grid = [['0' for col in range(m)] for row in range(n)]

        # for i in range(n):
        #     for j in range(m):
        #         grid[i][j] = Node((i, j))

        a = 3# n * m / 10
        p_terminal = random.randint(1, int(a))
        n_terminal = random.randint(1, int(a))
        barrier = random.randint(1, int(a))

        i = 0
        while i < p_terminal:
            row = random.randint(0, n - 1)
            col = random.randint(0, m - 1)
            if grid[row][col].tag is None:
                grid[row][col].terminal = random.randint(1, 9)
                self.grid[row][col] = grid[row][col].terminal
                grid[row][col].tag = "Terminal"
                i += 1

        i = 0
        while i < n_terminal:
            row = random.randint(0, n - 1)
            col = random.randint(0, m - 1)
            if grid[row][col].tag is None:
                grid[row][col].terminal = random.randint(-9, -1)
                self.grid[row][col] = grid[row][col].terminal
                grid[row][col].tag = "Terminal"
                i += 1

        i = 0
        while i < barrier:
            row = random.randint(0, n - 1)
            col = random.randint(0, m - 1)
            if grid[row][col].tag is None:
                grid[row][col].tag = "BARRIER"
                self.grid[row][col] = 'X'
                self.barrier.append([row, col])
                i += 1

        start_decision = random.choice([0, 1])

        if start_decision == 1:
            row = random.choice([0, n - 1])
            col = random.randint(0, m - 1)

        else:
            row = random.randint(0, n - 1)
            col = random.choice([0, m - 1])

        grid[row][col].tag = "START"
        self.grid[row][col] = 'S'
        self.start = [row, col]

        self.node_World = grid
        # return grid, [row, col]

    def takeAction(self, current_position, direction):
        obstacle= self.barrier
        grid_row_size = len(self.grid)
        grid_col_size = len(self.grid[0])
        new_position = copy.deepcopy(current_position)
        P = [self.prob_of_moving, (1-self.prob_of_moving)/2, (1-self.prob_of_moving)/2]

        # Choose an action according to the probability
        if direction == "up":
            action = np.random.choice(['up', '2up', 'down'], p= P)
        elif direction == "down":
            action = np.random.choice(['down', '2down', 'up'], p= P)
        elif direction == "left":
            action = np.random.choice(['left', '2left', 'right'], p= P)
        elif direction == "right":
            action = np.random.choice(['right', '2right', 'left'], p= P)
        else:
            print("Error: WRONG DIRECTION INPUT")

        # NOTE: BARRIER CHECK FOR SINGLE MOVE ACTIONS AND SECOND ACTION FOR DOUBLE MOVE ACTION IS DONE IN THE END

        # perform action
        if action == 'up':
            new_position[0] += -1
        if action == 'down':
            new_position[0] += 1
        if action == 'right':
            new_position[1] += 1
        if action == 'left':
            new_position[1] += -1
        if action == '2up':
            new_position[0] += -1
            barrier = new_position in obstacle or new_position[0] < 0 or new_position[0] >= grid_row_size or \
                      new_position[1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                current_position = copy.deepcopy(new_position)
                new_position[0] += -1
        elif action == '2down':
            new_position[0] += 1
            barrier = new_position in obstacle or new_position[0] < 0 or new_position[0] >= grid_row_size or \
                      new_position[1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                current_position = copy.deepcopy(new_position)
                new_position[0] += 1
        elif action == '2right':
            new_position[1] += 1
            barrier = new_position in obstacle or new_position[0] < 0 or new_position[0] >= grid_row_size or \
                      new_position[1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                current_position = copy.deepcopy(new_position)
                new_position[1] += 1
        elif action == '2left':
            new_position[1] += -1
            barrier = new_position in obstacle or new_position[0] < 0 or new_position[0] >= grid_row_size or \
                      new_position[1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                current_position = copy.deepcopy(new_position)
                new_position[1] += -1

        # barrier check for single move actions
        barrier = new_position in obstacle or new_position[0] < 0 or new_position[0] >= grid_row_size \
                  or new_position[1] < 0 or new_position[1] >= grid_col_size
        if not barrier:
            current_position = new_position

        return current_position

    def policy_random(self):
        actions = ["up", "down", "right", "left"]
        action = np.random.choice(actions)
        return action

    def policy_Epsilon_greedy(self, eps, current_node):
        p = np.random.random()
        if p <= eps:
            return self.policy_random()
        elif p > eps:
            return current_node.get_max_action()

    def policy_counter(self, current_node):
        actions = ["up", "down", "right", "left"]
        eps = 0.1
        new_actions = []
        for action in actions:
            if current_node.get_count(action) < self.min_count:
                new_actions.append(action)
        if len(new_actions) == 0:
            return self.policy_Epsilon_greedy(eps, current_node)
        else:
            return np.random.choice(new_actions)

    def direction_map(self, title = "direction map"):
        # Visualization of the found path using matplotlib
        fig, ax = plt.subplots(1)
        ax.margins()
        # Draw map
        row = len(self.node_World)  # map size
        col = len(self.node_World[0])  # map size
        for i in range(row):
            for j in range(col):
                x, y, state = self.node_World[i][j].get_print_action()
                avg_time = self.node_World[i][j].avg_timeOnNode(self.iter)
                if state == "BARRIER":
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='k', facecolor='k'))  # free space
                elif state == "START":
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='k', facecolor='b'))
                    ax.arrow(j - y, i - x, y, x, fc="k", ec="k", head_width=0.1,
                             head_length=0.1)
                    plt.annotate('Start', ha='center', va='bottom', xytext=(j, i-0.1), xy=(j, i-0.1))
                elif state >= 1:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='k', facecolor='g'))
                    plt.annotate(str(state), ha='center', va='bottom', xytext=(j, i-0.1), xy=(j, i-0.1))
                elif state <= -1:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='k', facecolor='r'))
                    plt.annotate(str(state), ha='center', va='bottom', xytext=(j, i-0.1), xy=(j, i-0.1))
                else:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='k', facecolor='w'))
                    if x != 0 or y != 0:
                        ax.arrow(j - y, i - x, y, x, fc="k", ec="k", head_width=0.1, head_length=0.1)
                        plt.annotate(str(avg_time), ha='center', va='bottom', xytext=(j, i-0.1), xy=(j, i-0.1))

        # Graph settings
        plt.title(title)
        plt.axis('scaled')
        plt.gca().invert_yaxis()
        plt.show()

    # def heat_map (self, title = "")

    def Q_learning(self):
        t_init = time.time()
        t_end = time.time()
        self.iter = 0

        while t_end - t_init <= self.time_to_learn:
            # restart from the start position
            current_node = self.node_World[self.start[0]][self.start[1]]
            first = True
            while first or (not current_node.terminal > 0 and t_end - t_init <= self.time_to_learn):
                action = self.policy_Epsilon_greedy(0.1, current_node)      # Exploration policy of the agent
                new_pos = self.takeAction([current_node.row, current_node.col], action)     # the action that agent actually took
                new_node = self.node_World[new_pos[0]][new_pos[1]]      # new node after the action we took

                #   Return max expected pay off for new state
                Q_ns_na = max([new_node.Q_up, new_node.Q_down, new_node.Q_right, new_node.Q_left])

                # expected pay of for current step for the action we took
                Q_s_a = current_node.get_Q_value(action)

                # update value calculation
                if not new_node.terminal == 0:
                    reward = new_node.terminal
                else:
                    reward = self.reward_per_action

                update_value = Q_s_a + self.alpha*(reward + (self.gamma*Q_ns_na) - Q_s_a)

                # update step
                current_node.update(action, update_value)
                current_node = new_node

                # debug
                # print(current_node.row, current_node.col)
                # if current_node.terminal < 0:
                #     print("PIT")
                #     # time.sleep(1)
                # elif current_node.terminal > 0:
                #     print("DONE")
                # print(current_node.tag)
                # debug
                first = False
                t_end = time.time()  # to cancel the sleep time

            self.iter += 1
            # print(iter)

    def SARSA(self):
        t_init = time.time()
        t_end = time.time()
        self.iter = 0

        while t_end - t_init <= self.time_to_learn:
            # restart from the start position
            current_node = self.node_World[self.start[0]][self.start[1]]
            action = self.policy_counter(current_node)  # Exploration policy of the agent
            current_node.counter_update(action)
            current_node.total_counter_update(self.iter)
            new_pos = self.takeAction([current_node.row, current_node.col], action)  # the action that agent actually took
            first = True
            while first or (not abs(current_node.terminal) > 0 and t_end - t_init <= self.time_to_learn):
                # new node after the action we took
                new_node = self.node_World[new_pos[0]][new_pos[1]]
                new_action = self.policy_counter(new_node)
                # new_node.counter_update(new_action)

                #   Return max expected pay off for new state
                Q_ns_na = new_node.get_Q_value(new_action)

                # expected pay of for current step for the action we took
                Q_s_a = current_node.get_Q_value(action)

                # update value calculation
                if not new_node.terminal == 0:
                    reward = new_node.terminal
                else:
                    reward = self.reward_per_action

                update_value = Q_s_a + self.alpha * (reward + (self.gamma * Q_ns_na) - Q_s_a)

                # update step

                new_node.total_counter_update(self.iter)
                current_node.update(action, update_value)
                current_node = new_node
                current_node.counter_update(action)
                action = new_action
                new_pos = self.takeAction([current_node.row, current_node.col], action)

                # debug
                # print(current_node.row, current_node.col)
                # if current_node.terminal < 0:
                #     print("PIT")
                #     # time.sleep(1)
                # elif current_node.terminal > 0:
                #     print("DONE")
                # print(current_node.tag)
                # debug
                first = False
                t_end = time.time()  # to cancel the sleep time

            self.iter += 1
            # print(iter)

if __name__ == '__main__':

    if len(sys.argv) != 6:
        print("Format: main.py <filename> <reward> <gamma> <time to learn> <movement probability>")
        exit(1)
    else:
        file = sys.argv[1]
        reward_per_action = float(sys.argv[2])
        gamma = float(sys.argv[3])
        time_to_learn = float(sys.argv[4])
        prob_of_moving = float(sys.argv[5])

        print("This program will read in", file)
        print("It will run for", time_to_learn, "seconds")
        print("Its decay rate is", gamma, "and the reward per action is", reward_per_action)
        print("Its transition model will move the agent properly with p =", prob_of_moving)

        grid_world = world(file, reward_per_action, gamma, time_to_learn, prob_of_moving)
        # grid_world.get_grid()
        grid_world.random_grid(40, 40)
        grid_world.SARSA()
        grid_world.direction_map()
        # print(grid_world.grid)