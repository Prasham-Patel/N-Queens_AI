import numpy as np
import random
import csv
import copy
import time

class CSV_:

    def __init__(self, path):
        self.CSVfilePath = path

    def creatCSV_board(self, board, cost, file_address=''):
        # default file address
        if len(file_address) == 0:
            file_address = self.CSVfilePath
        with open(file_address, 'a', newline='') as CSVfile:
            write = csv.writer(CSVfile)
            write.writerows(board)
            write.writerow([cost])
        CSVfile.close()

    def readCSV_board(self, file_address=''):

        # default file address
        if len(file_address) == 0:
            file_address = self.CSVfilePath

        grid = []

        with open(file_address, mode='r')as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                grid.append(lines)
        file.close()
        return grid

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


    def counter(self):
        pass

class world:

    def __init__(self, file, reward_per_action, gamma, time_to_learn, prob_of_moving):
        self.file = file
        self.reward_per_action = reward_per_action
        self.gamma = gamma
        self.time_to_learn = time_to_learn
        self.prob_of_moving = prob_of_moving
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

    def takeAction(self, current_position, direction):
        obstacle= self.barrier
        grid_row_size = len(self.grid)
        grid_col_size = len(self.grid[0])
        new_position = copy.deepcopy(current_position)
        P = [self.prob_of_moving, (1-self.prob_of_moving)/2, (1-self.prob_of_moving)/2]

        # Choose an action according to the probability
        if direction == "up":
            action = np.random.choice(['up', '2up', 'down'], p=[1, 0, 0])
        elif direction == "down":
            action = np.random.choice(['down', '2down', 'up'], p=[1, 0, 0])
        elif direction == "left":
            action = np.random.choice(['left', '2left', 'right'], p=[1, 0, 0])
        elif direction == "right":
            action = np.random.choice(['right', '2right', 'left'], p=[1, 0, 0])
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

    def Q_learning(self):
        t_init = time.time()
        t_end = time.time()
        print(self.start)
        iter = 0
        while t_end - t_init <= self.time_to_learn:
            # restart from the start position
            current_node = self.node_World[self.start[0]][self.start[1]]
            while not abs(current_node.terminal) > 0 and t_end - t_init <= self.time_to_learn:
                action = self.policy_Epsilon_greedy(0.01, current_node)      # Exploration policy of the agent
                new_pos = self.takeAction([current_node.row, current_node.col], action)     # the action that agent actually took
                new_node = self.node_World[new_pos[0]][new_pos[1]]      # new node after the action we took

                #   Return max expected pay off for new state
                Q_ns_na = max([new_node.Q_up, new_node.Q_down, new_node.Q_right, new_node.Q_left])

                # expected pay of for current step for the action we took
                Q_s_a = current_node.get_Q_value(action)

                # update value calculation
                update_value = Q_s_a + 0.2*(self.reward_per_action + (self.gamma*Q_ns_na) - Q_s_a)

                # update step
                current_node.update(action, update_value)
                current_node = new_node

                # debug
                # print(current_node.row, current_node.col)
                # if current_node.tag == "Terminal":
                #     print("DONE")
                #     time.sleep(1)
                # print(current_node.tag)
                # debug

                t_end = time.time()  # to cancel the sleep time

            iter += 1
            print(iter)

    def show_results(self):
        pass

if __name__ == '__main__':

    grid_world = world("F:\ML and CV\WPI_AI\Assignment_3\grid_sample.txt", -0.01, 0.9, 5, 1)
    grid_world.get_grid()
    grid_world.Q_learning()
