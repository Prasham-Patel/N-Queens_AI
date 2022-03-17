import numpy as np
import csv
import copy

def print_board(board):
    for rows in board:
        print(rows)

def boardGenerator(size = 8, Heavy_Queen = True, num_queens = 0):

    # the board with < 4 queens do not have a solution for Heavy Queens problem
    if size < 4 and Heavy_Queen:
        print("size of the board must be GREATER than 4")
        return None

    board = [[0 for j in range(size)] for i in range(size)]
    queen_pos = []

    if Heavy_Queen:
        for i in range(size):
            pos = np.random.randint(low=0, high=size)
            board[pos][i] = np.random.randint(low=1, high=10)
            queen_pos.append([pos, i])

    if not Heavy_Queen:
        if num_queens < size:
            print("for 9N/8 problem, number of queens must be greater than length of side of the board")
            return None
        for i in range(size):
            pos = np.random.randint(low=0, high=size)
            board[pos][i] = np.random.randint(low=1, high=10)
            queen_pos.append([pos, i])
        i = 0
        while i != num_queens - size:
            row = np.random.randint(low=0, high=size)
            col = np.random.randint(low = 0, high=size)
            if board[row][col] != 0:
                continue
            else:
                board[row][col] = np.random.randint(low=1, high=10)
                queen_pos.append([row, col])
                i += 1
    return board

def getQueenPos(board):
    queen_pos = []
    queen_weight = []
    for j in range(len(board[0])):
        for i in range(len(board)):
            if int(board[i][j]) != 0:
                queen_pos.append([i, j])
                queen_weight.append(board[i][j])

    return queen_pos, queen_weight

def creatCSV(board):
    with open('board', 'w',newline='') as CSVfile:
        write = csv.writer(CSVfile)
        write.writerows(board)

def readCSV(file_address = ""):

    # default file address
    if len(file_address) == 0:
        file_address = "F:\ML and CV\WPI_AI\Assignment_1\\board"

    board = []

    with open(file_address, mode='r')as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            board.append(list(lines))

        for rows in range(len(board)):
            print(board[rows])

    file.close()
    return board

def is_attacking(board, pos):
    attacks = 0
    attackers = []
    print("pos",pos)

    if len(board) == 0:
        return None

    for i in range(len(board[0])):
        print("index i" ,i)
        if int(board[pos[0]][i]) != -1 and i != pos[1]:
            print(pos[0], i)
            attackers.append([pos[0], i])
            attacks += 1


    i = pos[0] + 1
    j = pos[1] + 1
    while i < len(board) and j < len(board[0]):
        if int(board[i][j]) != -1:
            print(j, i)
            attackers.append([i, j])
            attacks += 1
        i += 1
        j += 1

    i = pos[0] - 1
    j = pos[1] - 1
    while i >= 0 and j >= 0:
        if int(board[i][j]) != -1:
            print(i, j)
            attackers.append([i, j])
            attacks += 1
        i -= 1
        j -= 1

    i = pos[0] - 1
    j = pos[1] + 1
    while i >= 0 and j < len(board[0]):
        if int(board[i][j]) != -1:
            print(i, j)
            attackers.append([i, j])
            attacks += 1
        i -= 1
        j += 1

    i = pos[0] + 1
    j = pos[1] - 1
    while i > len(board) and j >= 0:
        if int(board[i][j]) != -1:
            print(i, j)
            attackers.append([i, j])
            attacks += 1
        i += 1
        j -= 1

    print("done")
    print(attacks)
    print("attackers", attackers)

    return attacks, attackers

def total_attacks(board, queen_pos):
    board_copy = copy.deepcopy(board)
    attacks = 0
    attackers = []
    iter = 0
    for queen in queen_pos:
        new_attacks, new_attackers = is_attacking(board_copy, [queen[0], queen[1]-iter])
        attacks += new_attacks
        attackers = attackers + new_attackers
        [j.pop(0) for j in board_copy]
        iter += 1

    print("number of attacks",attacks)
    print(attackers)


def conv_to_string(queen_pos):
    string = ""
    for i in range(len(queen_pos)):
        string += str(queen_pos[i][0])
    return string


if __name__ == "__main__":
    creatCSV(boardGenerator(size=8))
    board = readCSV()
    queen_pos, weight = getQueenPos(board)
    print("queen pos",queen_pos)
    # new_board = [j.pop(0) for j in board]
    # for rows in range(len(board)):
    #     print(board[rows])
    # total_attacks(board, queen_pos)

    string = ""
    for i in range(len(queen_pos)):
        string += str(queen_pos[i][0])

    print(type(string))

