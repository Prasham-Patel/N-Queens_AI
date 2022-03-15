import numpy as np
import random
import copy
import csv

#function to read the csv file by providing the address of the file


                                             #input the grid size
def random_pattern(n):
    queens = []                                              #make an array of wighted queens
    for i in range(0, n): 
        r = random.randint(1,9)                              #generate a random no. between 0 and 9
        queens.append(r) 
    # print(queens)                                            #add that random no. in weighted queens array
    start_pattern = np.zeros((n, n), dtype = np.int64)       #make a matrix of nxn grid(all elements initialised to zero)
    for i in range(0, n):                                    #put weighted queens randomly in each column
        r = random.randint(0,n-1)       
        start_pattern[r][i] = queens[i]
    start_pattern = start_pattern.tolist() 
    return start_pattern, queens
        

def positions_moved(grid, start_pattern,queens):
    position_start = []
    position_current = []

    for i in range(0,n):
        for j in range(0,n):
            if start_pattern[j][i] != 0:
                position_start.append(j)
            if grid[j][i] != 0:
                position_current.append(j)
    h = 0
    for i in range(0,n):
        h = h + abs(position_start[i] - position_current[i]) * (queens[i]*queens[i])
    return h

def getQueenPos(board):
    queen_pos = []
    queen_weight = []
    for j in range(len(board[0])):
        for i in range(len(board)):
            if int(board[i][j]) != 0:
#                 print(board[i][j])
                queen_pos.append([i, j])
                queen_weight.append(board[i][j])

    return queen_pos, queen_weight

def is_attacking(board, pos):
    attacks = 0
    attackers = []
#     print("pos",pos)

    if len(board) == 0:
        return None

    for i in range(len(board[0])):
#         print("index i" ,i)
        if int(board[pos[0]][i]) != 0 and i != pos[1]:
#             print(pos[0], i)
            attackers.append([pos[0], i])
            attacks += 1


    i = pos[0] + 1
    j = pos[1] + 1
    while i < len(board) and j < len(board[0]):
        if int(board[i][j]) != 0:
#             print(j, i)
            attackers.append([i, j])
            attacks += 1
        i += 1
        j += 1

    i = pos[0] - 1
    j = pos[1] - 1
    while i >= 0 and j >= 0:
        if int(board[i][j]) != 0:
#             print(i, j)
            attackers.append([i, j])
            attacks += 1
        i -= 1
        j -= 1

    i = pos[0] - 1
    j = pos[1] + 1
    while i >= 0 and j < len(board[0]):
        if int(board[i][j]) != 0:
#             print(i, j)
            attackers.append([i, j])
            attacks += 1
        i -= 1
        j += 1

    i = pos[0] + 1
    j = pos[1] - 1
    while i > len(board) and j >= 0:
        if int(board[i][j]) != 0:
#             print(i, j)
            attackers.append([i, j])
            attacks += 1
        i += 1
        j -= 1

#     print("done")1
#     print(attacks)
#     print("attackers", attackers)

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


#     print("number of attacks",attacks)
#     print(attackers)

def heuristic1(grid):
    queenPos, queenWeight = getQueenPos(grid)
    he = 0
#     print(queenPos)
    for i in range(len(queenPos)):
        attacks, attacker = is_attacking(grid, queenPos[i])
        he += queenWeight[i]**2*attacks
#         print(he)
    return he
# heuristic1(start_pattern)


class Node:
    
    def __init__(self, grid, parent=None):
        self.grid = grid
        self.parent = parent
        self.cost = heuristic1(self.grid)
        self.g = 0
        self.f = 0
        

def astar(start_pattern,queens):
    start = Node(start_pattern, None)                    #initialise start node
    start.parent = start 
    open_list = []                                       
    closed_list = []                                     #list of visited nodes
    open_list.append(start)
    # print("Heuristic of start pattern: ", start.cost,"\n")#append open list with start node
    # print("Start pattern\n")
    # for line in start.grid:
    #     print ('  '.join(map(str, line)))
    # print("\n")
    while len(open_list)>0:
        
        current_pattern = open_list[0]
        current_index = 0
        
        for index, item in enumerate(open_list):         #find the node with minimum cost out of all childs in open_list
            if item.f < current_pattern.f:
                current_pattern = item
                current_index = index
                
        closed_list.append(current_pattern)              #add the node with least cost in closed_list
        open_list.pop(current_index)                     #pop the visited node from the open_list
                     
        #print(current_pattern.grid)
#         for line in current_pattern.grid:
#             print ('  '.join(map(str, line)))               #for debugging
#         print("\n\n")
        
        if current_pattern.cost == 0:                    #returns the path taken to reach desired state
            return current_pattern.grid, current_pattern.g
        
        children = []                                    #returns all the child nodes for the current node
        for i in range(0,n):
            for j in range(0,n):
                if current_pattern.grid[i][j] != 0:
                    for k in range(0,n):
                        if k!=i:
                            newgrid = copy.deepcopy(current_pattern.grid)
                            temp = newgrid[i][j]
                            newgrid[i][j] = 0
                            newgrid[k][j] = temp                                
                            new_pattern = Node(newgrid, current_pattern)
                            new_pattern.g = positions_moved(new_pattern.grid,start_pattern,queens)
                            new_pattern.f = new_pattern.cost + new_pattern.g
                            children.append(new_pattern)
                            
        for child in children:                     #remove all child nodes which are already visited i.e. in closed_list
            
            flag = False
            for item in closed_list:
                if child.grid == item.grid:
                    flag = True
                    break
            if flag == True:
                continue
            for open_node in open_list:
                if child.grid == open_node.grid and child.g >= open_node.g:
                    flag = True
                    break
            if flag == False:
                open_list.append(child)


def queen_tuple(grid):
    queen_board = []
    for i in range(0,len(grid)):
        for j in range(0,len(grid)):
            if grid[j][i] !=0:
                queen_board.append((j,grid[j][i]))
                continue
    return queen_board

class training_sample:

#define attributes of the training samples we need i.e. initial pattern, solved pattern and solved pattern cost by astar search algorithm.

    def __init__(self):
        self.pattern, self.queens = random_pattern(n)
        self.solved_pattern, self.astar_cost = astar(self.pattern, self.queens)

# training_set_generation:

n = 5                                   # grid size you want to solve
samples = 4                             # no. of training samples you want to generate
training_set = []                       # List of all training samples
for i in range(0,samples):
    start = training_sample()
    training_set.append(start)
    # for line in start.pattern:
    #     print ('  '.join(map(str, line)))               #for debugging
    # print("\n\n")
    # for line in start.solved_pattern:
    #     print ('  '.join(map(str, line)))               #for debugging
    # print("\n\n")
    # print("Cost",start.astar_cost)
    # print("\n\n")



