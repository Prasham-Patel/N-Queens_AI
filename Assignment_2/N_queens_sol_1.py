import numpy as np
import csv
import Utility as utility
import os
from math import inf


class N_queens:

    def __init__(self, N):
        self.numberOfQueens = N
        self.results = []

    ''' Python3 program to solve N Queen Problem using
    backtracking '''

    ''' A utility function to check if a queen can
    be placed on board[row][col]. Note that this
    function is called when "col" queens are
    already placed in columns from 0 to col -1.
    So we need to check only left side for
    attacking queens '''

    def isSafe(self, board, row, col):
        # Check this row on left side
        for i in range(col):
            if (board[row][i]):
                return False

        # Check upper diagonal on left side
        i = row
        j = col
        while i >= 0 and j >= 0:
            if (board[i][j]):
                return False
            i -= 1
            j -= 1

        # Check lower diagonal on left side
        i = row
        j = col
        while j >= 0 and i < len(board):
            if (board[i][j]):
                return False
            i = i + 1
            j = j - 1

        return True

    ''' A recursive utility function to solve N
    Queen problem '''

    def solveNQUtil(self, board, col):
        ''' base case: If all queens are placed
        then return true '''
        if (col == len(board)):
            v = []
            for i in board:
                for j in range(len(i)):
                    if i[j] == 1:
                        v.append(j + 1)
            self.results.append(v)
            return True

        ''' Consider this column and try placing
        this queen in all rows one by one '''
        res = False
        for i in range(len(board)):

            ''' Check if queen can be placed on
            board[i][col] '''
            if (self.isSafe(board, i, col)):
                # Place this queen in board[i][col]
                board[i][col] = 1

                # Make result true if any placement
                # is possible
                res = self.solveNQUtil(board, col + 1) or res

                ''' If placing queen in board[i][col]
                doesn't lead to a solution, then
                remove queen from board[i][col] '''
                board[i][col] = 0  # BACKTRACK

        ''' If queen can not be place in any row in
            this column col then return false '''
        return res

    ''' This function solves the N Queen problem using
    Backtracking. It mainly uses solveNQUtil() to
    solve the problem. It returns false if queens
    cannot be placed, otherwise return true and
    prints placement of queens in the form of 1s.
    Please note that there may be more than one
    solutions, this function prints one of the
    feasible solutions.'''

    def solveNQ(self):
        self.results.clear()
        board = [[0 for j in range(self.numberOfQueens)]
                 for i in range(self.numberOfQueens)]
        self.solveNQUtil(board, 0)
        self.results.sort()


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

        board_list = []
        cost_list = []

        with open(file_address, mode='r')as file:
            csvFile = csv.reader(file)
            board = []
            board_size = int(next(csvFile)[0])  # first line is the length of board
            index = 0
            for lines in csvFile:

                # only when there are blank lines
                if index == 0:
                    board.clear()  # new board will start after cost
                    # continue

                if index < board_size:
                    board.append(list(lines))
                    index += 1
                    continue

                if index == board_size:
                    board_list.append(board)
                    cost_list.append(int(lines[0]))
                    index = 0
                    # print(board)
                    
                    continue

        file.close()
        return board_list, cost_list

    def read_sols(self, file_address=''):
        if len(file_address) == 0:
            file_address = self.CSVfilePath
        queen_list = []
        with open(file_address, mode='r')as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                queen_list.append([int(i) for i in list(lines)])
        file.close()
        return queen_list

    def write_sols(self, queen_list, file_address=''):
        if len(file_address) == 0:
            file_address = self.CSVfilePath
        with open(file_address, 'w', newline='') as CSVfile:
            write = csv.writer(CSVfile)
            write.writerows(queen_list)
        CSVfile.close()


# Driver Code

counter = 0
while counter < 50000:
    print(counter)
    n = 5
    solver = N_queens(n)
    solver.solveNQ()
    res = solver.results
    dir_list = os.listdir("F:\Study\Artificial Intelligence - CS 534\Assignments\Assignment_2")
    #dir_list = os.listdir("N_queens_sols")

    # create 4 files to store the solutions | run this code only once to initialize file
    #
    # for i in range(1, 4):
    #     file = "N_queens_sols/N" + str(4 + i) + "_sol.txt"
    #     with open(file, 'w') as fp:
    #         pass
    #     fp.close()
    csv_ = CSV_("F:\Study\Artificial Intelligence - CS 534\Assignments\Assignment_2\\N_queens_sols\\N8_sol.txt")
    csv_.write_sols(res)
    new_list = csv_.read_sols()
    board = utility.boardGenerator(n)
    queen_pos, queen_weight = utility.getQueenPos(board)
    cost = inf

    for boards in new_list:
        new_cost = 0
        for i in range(len(queen_pos)):
            new_cost += abs(int(queen_pos[i][0]) - int(boards[i]-1)) * (queen_weight[i]**2) # -1 ADDED AS INDEX STARTS FROM 1 RATHER THAN 0
        if new_cost < cost:
            cost = new_cost
    csv_.creatCSV_board(board, cost, "F:\Study\Artificial Intelligence - CS 534\Assignments\Assignment_2\\Data\\N5.txt")
    counter+=1

# create 4 files to store the solutions | only run this code once

# for i in range(1, 5):
#     file = "F:\Study\Artificial Intelligence - CS 534\Assignments\Assignment_2\\Data\\Data/N" + str(4 + i) + ".txt"
#     with open(file, 'w', newline='') as fp:
#         write = csv.writer(fp)
#         write.writerow([4+i])
#     fp.close()

csv_ = CSV_("F:\Study\Artificial Intelligence - CS 534\Assignments\Assignment_2\\Data\\N5.txt")
board_list, cost_list = csv_.readCSV_board("F:\Study\Artificial Intelligence - CS 534\Assignments\Assignment_2\\Data\\N5.txt")
print(cost_list)
# for line in board_list[0]:
#     print(line)
# # print(board_list[])
