import numpy as np
import random
import copy
import math
import time
import csv

#function to generate a random nxn grid given the weighted queens and their respective columns
def random_pattern(queens):
    start_pattern = np.zeros((n, n), dtype = np.int64)
    for i in range(0, n):                                      
        r = random.randint(0,n-1)       
        start_pattern[r][i] = queens[i]
    
    for i in range(0,q):
        while True:
            a = random.randint(0,n-1)
            if start_pattern[a][b[i]] !=0:
                continue
            start_pattern[a][b[i]] = queens[n+i]
            break
    start_pattern = start_pattern.tolist()
    return start_pattern

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

#select your way of input
print("Please select a method out of below options to generate nxn heavy queens grid\n1. Generate a random nxn grid\n2. Input a csv file\nType 1 or 2\n")
method = input()
if method == '1':
    print("Enter the grid size, n : ")
    n = input()                                              #input the grid size
    n = int(n)
    q = int(n/8)
    total_queens = n + q
    queens = []                                              #make an array of wighted queens
    for i in range(0, n+q):
        r = random.randint(1,9)                              #generate a random no. between 0 and 9
        queens.append(r)  

    b=[]                                                     #list to keep random generated column for extra queens
    for i in range(0,q):
        b1 = random.randint(0,n-1)
        b.append(b1)

    start_pattern = random_pattern(queens)
    for line in start_pattern:
        print ('  '.join(map(str, line)))
    
elif method == '2':
    print("Please enter the address of your csv file : ")
    address = input()
    start_pattern = readCSV(file_address = address)         #Read the csv file
    n=len(start_pattern)                                    #calculate the grid length value
    queens = []                                             #list of queen weights in respective columns
    b = []                                                  #list to keep random generated column for extra queens
    for i in range(0, len(start_pattern)):
        count = 0
        for j in range(0, len(start_pattern)):
            start_pattern[j][i] = int(start_pattern[j][i])
            if start_pattern[j][i] !=0:
                queens.append(start_pattern[j][i])  
                count+=1
        if count >1:
            b.append(i)
    total_queens = len(queens)
    q = total_queens - n
else:
    print("Invalid entry")

#calculates the cost of movement w.r.t initial grid ( = (wt. of queen)^2 * no. of steps moved)

def positions_moved(grid):
    position_start = []
    position_current = []
    global start_pattern
    global total_queens
    global queens
    for i in range(0,n):
        for j in range(0,n):
            if start_pattern[j][i] != 0:
                position_start.append(j)
            if grid[j][i] != 0:
                position_current.append(j)
    h = 0
    for i in range(0,total_queens):
        h = h + abs((position_start[i] - position_current[i]) * queens[i]*queens[i])
    return h

#calculates no. of attacking queens (directly or indirectly) on the grid

def attack_queens(grid):     
        attack1 = 0                             #horizontal and vertical attacks
        attack2 = 0                             #diagonal attacks
        for i in range(0,n):
            for j in range(0,n):
                if grid[i][j] != 0:
                    attack1 -= 2
                    for k in range(0,n):
                        if grid[i][k] != 0:
                            attack1 += 1
                        if grid[k][j] != 0:
                            attack1 += 1
                            
                    k, l = i+1, j+1
                    while k < n and l < n:
                        if grid[k][l] != 0:
                            attack2 += 1
                        k +=1
                        l +=1
                    k, l = i+1, j-1
                    while k < n and l >= 0:
                        if grid[k][l] != 0:
                            attack2 += 1
                        k +=1
                        l -=1
                    k, l = i-1, j+1
                    while k >= 0 and l < n:
                        if grid[k][l] != 0:
                            attack2 += 1
                        k -=1
                        l +=1
                    k, l = i-1, j-1
                    while k >= 0 and l >= 0:
                        if grid[k][l] != 0:
                            attack2 += 1
                        k -=1
                        l -=1
        attacking_queens = ((attack2 + attack1)/2)
        total_cost = 100*attacking_queens                       
        return total_cost

class Node:
    
    def __init__(self, grid, parent=None):
        self.grid = grid
        self.parent = parent
        self.cost = attack_queens(self.grid)
        self.total_cost = self.cost  + positions_moved(self.grid)      #calculates cost function w.r.t initial pattern (= 100*attacking queens + movement cost w.r.t initial pattern)
    

def hill_climb(start_pattern):
    global queens
    start = Node(start_pattern, None)                    #initialise start node
    start.parent = start 
    print("Cost of start pattern : ",start.total_cost,"\n\n")
    open_list = []                                      #keeps all neighbours of a current node
    closed_list = []                                    #keeps all the visited nodes
    open_list.append(start)                             #append open list with start node
    counter = 0                                         #counts no. of steps before a random restart
    temperature = 100*n                                 #temperature initialisation for simulated annealing
    probability = 0
    costs = []                                          #list to store all visited neighbour's cost
    min_cost_10 = []                                    #list to store minimum cost every 10 seconds(or given interval)
    start_time = time.time()                            #start the timer
    stop_seconds = 120                                  #time(in seconds) for the hill climb algorithm to run
    
    while time.time()-start_time < stop_seconds:
        start_time1 = time.time()
        stop_seconds1 = 10
        min_list = []
        while time.time() - start_time1 < stop_seconds1:
            print("counter : ",counter,"\n")

            counter = counter + 1
            temperature = temperature/math.log(counter+1000,1000)   #temperature decreasing function
            current_pattern = open_list[0]
            current_index = 0

            for index, item in enumerate(open_list):         #find the node with minimum cost out of all childs in open_list
                if item.total_cost <= current_pattern.total_cost:
                    current_pattern = item
                    current_index = index

            #Simulated Annealing
            
            probability = 1
            if current_pattern.total_cost <= current_pattern.parent.total_cost:
                probability = 1
            else:
                #probability function
                probability = math.exp((-current_pattern.total_cost + current_pattern.parent.total_cost)/temperature)
            print("Probability to move on next neighbour : ",probability,"\n")

            if probability < 0.95:                           #random restart condition after simulated annealing
                print("Random restart done after probability : ", probability,"\n")
                for i in range(len(open_list),0,-1):             
                    open_list.pop(i-1)
                print("random\n")
                temperature = 100*n
                counter = 0
                random_start = Node(random_pattern(queens), None)
                random_start.parent = random_start
                open_list.append(random_start)
                continue
          
            closed_list.append(current_pattern)              #add the node with least cost in closed_list
            costs.append(current_pattern.total_cost)
            min_list.append(current_pattern.total_cost)
            
            for i in range(len(open_list),0,-1):             #remove all nodes from open_list (greedy search algo)
                open_list.pop(i-1)

            for line in current_pattern.grid:
                print ('  '.join(map(str, line)))               #for debugging
            print("\n") 
            print("Total cost : ", current_pattern.total_cost,"\n\n")


            children = []                                    #returns all the child nodes for the current node
            for i in range(0,n):
                for j in range(0,n):
                    if current_pattern.grid[i][j] != 0:
                        for k in range(0,n):
                            if k!=i:
                                newgrid = copy.deepcopy(current_pattern.grid)
                                temp = newgrid[i][j]
                                newgrid[i][j] = 0
                                if newgrid[k][j] == 0:
                                    newgrid[k][j] = temp                                
                                    new_pattern = Node(newgrid, current_pattern)
                                    children.append(new_pattern)

            for child in children:                         #remove all child nodes which are already visited i.e. in closed_list
                flag = False
                for item in closed_list:
                    if (item.grid == child.grid):
                        flag = True
                        break
                if flag == False:
                    open_list.append(child)

        min_cost_10.append(min(min_list))                 
    return min_cost_10                                     #gives least cost grid after every 10 seconds

print("Start Pattern : \n")
for line in start_pattern:
    print ('  '.join(map(str, line)))
print("\n")
min_cost = hill_climb(start_pattern)
print("Results of Hill climbing-----------------------------\n")
print("Minimum cost after every 10 seconds : ", min_cost)
print("\n")
print("Overall minimum cost is ", min(min_cost))