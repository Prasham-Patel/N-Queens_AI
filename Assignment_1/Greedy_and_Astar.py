import numpy as np
import random
import copy
import csv

#function to read the csv file by providing the address of the file
def readCSV(file_address = ""):

    # default file address
    if len(file_address) == 0:
        file_address = "F:\ML and CV\WPI_AI\Assignment_1\\board"

    board = []

    with open(file_address, mode='r',encoding='utf-8-sig')as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            board.append(list(lines))

        for rows in range(len(board)):
            print(board[rows])

    file.close()
    return board

print("Please select a method out of below options to generate nxn heavy queens grid\n1. Generate a random nxn grid\n2. Input a csv file\nType 1 or 2\n")
method = input()
if method == '1':
    print("Enter the grid size, n : ")
    n = input()                                              #input the grid size
    n = int(n)
    queens = []                                              #make an array of wighted queens
    for i in range(0, n): 
        r = random.randint(1,9)                              #generate a random no. between o and 9
        queens.append(r) 
    # print(queens)                                            #add that random no. in weighted queens array
    start_pattern = np.zeros((n, n), dtype = np.int64)       #make a matrix of nxn grid(all elements initialised to zero)
    for i in range(0, n):                                    #put weighted queens randomly in each column
        r = random.randint(0,n-1)       
        start_pattern[r][i] = queens[i]
    start_pattern = start_pattern.tolist() 
        
elif method == '2':
    print("Please enter the address of your csv file : ")
    address = input()
    start_pattern = readCSV(file_address = address)         #Read the csv file
    n=len(start_pattern)                                    #calculate the grid length value
    queens = []                                             #list of queen weights in respective columns
    for i in range(0, len(start_pattern)):
        for j in range(0, len(start_pattern)):    
            start_pattern[j][i] = int(start_pattern[j][i])
            if start_pattern[j][i] !=0:
                queens.append(start_pattern[j][i])    
else:
    print("Invalid entry")

def positions_moved(grid):
    position_start = []
    position_current = []
    global start_pattern
    global queens
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

#     print("done")
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
        

def greedy_search(start_pattern):
    
    start = Node(start_pattern, None)                    #initialise start node
    start.parent = start 
    open_list = []
    closed_list = []
    open_list.append(start)                              #append open list with start node
    # print("Heuristic of start pattern: ", start.cost,"\n")
    print("Start pattern\n")
    for line in start.grid:
        print ('  '.join(map(str, line)))
    print("\n")

    while len(open_list)>0:
        
        current_pattern = open_list[0]
        current_index = 0
        
        for index, item in enumerate(open_list):         #find the node with minimum cost out of all childs in open_list
            if item.cost < current_pattern.cost:
                current_pattern = item
                current_index = index
                
        closed_list.append(current_pattern)              #add the node with least cost in closed_list
        for i in range(len(open_list),0,-1):             #remove all nodes from open_list (greedy search algo)
            open_list.pop(i-1)
                     
        #print(current_pattern.grid)
#         for line in current_pattern.grid:
#             print ('  '.join(map(str, line)))               #for debugging
#         print("\n\n")
                
        if current_pattern.cost == 0:                    #returns the path taken to reach desired state
            path = []
            costs = []
            current = current_pattern
            while current is not start:
                path.append(current.grid)
                costs.append(current.g)
                current = current.parent
            return path[::-1],costs[::-1] 
        
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
                            new_pattern.g = positions_moved(new_pattern.grid)
                            children.append(new_pattern)
                            
        for child in children:                     #remove all child nodes which are already visited i.e. in closed_list
            flag = False
            for item in closed_list:
                if (item.grid == child.grid):
                    flag = True
                    break
            if flag == False:
                open_list.append(child)        
                
def astar(start_pattern):
    start = Node(start_pattern, None)                    #initialise start node
    start.parent = start 
    open_list = []                                       
    closed_list = []                                     #list of visited nodes
    open_list.append(start)
    # print("Heuristic of start pattern: ", start.cost,"\n")#append open list with start node
    print("Start pattern\n")
    for line in start.grid:
        print ('  '.join(map(str, line)))
    print("\n")
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
            path = []
            costs = []
            current = current_pattern
            while current is not start:
                path.append(current.grid)
                costs.append(current.g)
                current = current.parent
            return path[::-1], costs[::-1]
        
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
                            new_pattern.g = positions_moved(new_pattern.grid)
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

print("Using Greedy search ------------------\n\n")
path,cost = greedy_search(start_pattern)
for i in range(0,len(path)):
    print("Move : ",i+1,"    Cost : ",cost[i],"\n")
    for line in path[i]:
        print ('  '.join(map(str, line)))
    print("\n\n")

print("Using A* search ------------------\n\n")
path1, cost1 = astar(start_pattern)
for i in range(0,len(path1)):
    print("Move : ",i+1,"    Cost1 : ",cost1[i],"\n")
    for line in path1[i]:
        print ('  '.join(map(str, line)))
    print("\n\n")
print("Using Greedy search: Reached optimal solution in {} moves with a cost of {}".format(len(path),cost[len(path)-1]))
print("Using A* search    : Reached optimal solution in {} moves with a cost of {}".format(len(path1),cost[len(path1)-1]))
