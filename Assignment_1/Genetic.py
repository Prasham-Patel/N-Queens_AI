import numpy as np
import random
import copy
import math
import time
import csv


def random_populate(grid):
    new_grid = copy.deepcopy(grid)
    numChange = int(len(new_grid) * 0.5)  # Randomly changes 0.25 rows

    i = 0
    while i < numChange:
        rand_col = random.randint(0, len(new_grid) - 1)
        # print(i)

        done = False

        for row in range(len(new_grid[:][rand_col])):
            if new_grid[row][rand_col] > 0:
                # print(row, rand_col)
                rand_row = random.randint(0, len(new_grid) - 1)
                if (new_grid[rand_row][rand_col] == 0):
                    new_grid[rand_row][rand_col] = new_grid[row][rand_col]
                    new_grid[row][rand_col] = 0
                    done = True
                    break

        if done:
            i += 1

    # for line in new_grid:
    #   print ('  '.join(map(str, line)))

    return new_grid


def positions_moved(grid):
    position_start = []
    position_current = []
    global start_pattern
    global total_queens
    global queens
    global n
    for i in range(0, n):
        for j in range(0, n):
            if start_pattern[j][i] != 0:
                position_start.append(j)
            if grid[j][i] != 0:
                position_current.append(j)
    h = 0
    for i in range(0, total_queens):
        h = h + abs((position_start[i] - position_current[i]) * queens[i] * queens[i])

    return h

def calc_cost(grid):     #calculates no. of attacking queens
        totalhcost = 0
        totaldcost = 0
        for i in range(0,n):
            for j in range(0,n):
                #if this node is a queen, calculate all violations
                if grid[i][j] != 0:
                #subtract 2 so don't count self
                #sideways and vertical
                    totalhcost -= 2
                    for k in range(0,n):
                        if grid[i][k] != 0:
                            totalhcost += 1
                        if grid[k][j] != 0:
                            totalhcost += 1
                  #calculate diagonal violations
                    k, l = i+1, j+1
                    while k < n and l < n:
                        if grid[k][l] != 0:
                            totaldcost += 1
                        k +=1
                        l +=1
                    k, l = i+1, j-1
                    while k < n and l >= 0:
                        if grid[k][l] != 0:
                            totaldcost += 1
                        k +=1
                        l -=1
                    k, l = i-1, j+1
                    while k >= 0 and l < n:
                        if grid[k][l] != 0:
                            totaldcost += 1
                        k -=1
                        l +=1
                    k, l = i-1, j-1
                    while k >= 0 and l >= 0:
                        if grid[k][l] != 0:
                            totaldcost += 1
                        k -=1
                        l -=1
        attacking_queens = ((totaldcost + totalhcost)/2)
        total_cost = 100*attacking_queens
        return total_cost


def cost_of_grid(grid):
  cost = positions_moved(grid) + calc_cost(grid)
  # print(cost)
  return cost


def generate_child(parent1, parent2):
    mother = copy.deepcopy(parent1)
    father = copy.deepcopy(parent2)

    mother = np.transpose(parent1[0])
    father = np.transpose(parent2[0])

    mid_line = random.randint(0, len(mother) - 1)

    child1 = []
    i = 0
    while i < len(mother):
        if i < mid_line:
            child1.append(mother[i])

        else:
            child1.append(father[i])
        i = i + 1

    child1 = np.transpose(child1)

    child2 = []
    i = 0
    while i < len(mother):
        if i < mid_line:
            child2.append(father[i])

        else:
            child2.append(mother[i])
        i = i + 1

    child2 = np.transpose(child2)

    if (cost_of_grid(child1) > cost_of_grid(parent1[0])) or (cost_of_grid(child1) > cost_of_grid(parent2[0])):
        fit_1 = False
    else:
        fit_1 = True

    if cost_of_grid(child2) > cost_of_grid(parent1[0]) or cost_of_grid(child2) > cost_of_grid(parent2[0]):
        fit_2 = False
    else:
        fit_2 = True

    return [child1, cost_of_grid(child1)], [child2, cost_of_grid(child2)], fit_1, fit_2


def first_gen(grid, n=100):
    gen = [[grid, cost_of_grid(grid)]]

    i = 1

    while i < n:
        new_grid = random_populate(grid)
        gen.append([new_grid, cost_of_grid(new_grid)])
        i += 1
        # print(i)

    return gen


def get_rand_parents(gen, max_weight, weights):
    # gen must be sorted
    # gen = np.asarray(gen)

    # weights = []

    # for line in gen:
    #   weights.append((max_weight - line[1])**1.5)

    # max

    return random.choices(gen, weights, k=2)


from hashlib import new


def get_new_gen(gen, n=100):
    # gen = copy.deepcopy(gen)
    # print(gen)
    # gen = sorted(gen, key= lambda x: x[1]) # give already sorted array as an input

    new_gen = []

    gen_copy = copy.deepcopy(gen)

    elites = gen_copy[:int(len(gen) * 0.1)]

    max_weight = gen_copy[-1][1]

    i = 0
    while i < (len(elites)):  # Selection
        new_gen.append(elites[i])
        i += 1

        # prasham's weighted mutattion

    mutate = random.choices([True, False], [0.2, 0.8], k=1)
    if mutate == True:  # muttation
        print("mutation", mutate)
        i = 0
        while i < len(gen) * 0.1 or i < 1:
            j = random.randint(0, len(gen) - 1)
            mutate = random_populate(gen[j][0])
            # print(mutate)
            mutate = [mutate, cost_of_grid(mutate)]
            # if mutate not in new_gen:
            if mutate[1] < gen[j][1]:
                # new_gen.append(mutate)
                new_gen.append(mutate)
            i += 1

    # mutate in elites only

    # if random.choices([True, False], [0.5, 0], k=1):
    #   i = 0
    #   while i < len(new_gen)*0.1:
    #     mutate = random_populate(elites[random.randint(0, len(elites)-1)][0])
    #     # print(mutate)
    #     mutate = [mutate, cost_of_grid(mutate)]
    #     # if mutate not in new_gen:
    #     new_gen.append(mutate)
    #     i += 1

    # purna's high muttation rate code

    # i = 0
    # while i < (len(gen)*0.15):                # Mutation
    #   mutate = random_populate(gen[i][0])
    #   # print(mutate)
    #   mutate = [mutate, cost_of_grid(mutate)]
    #   # if mutate not in new_gen:
    #   if mutate[1] < gen[i][1]:
    #     new_gen.append(mutate)
    #   i +=1

    remaining_gen = gen_copy[:int(len(gen) * 0.7)]  # culling

    weights = []  # probability of parent selection

    for line in remaining_gen:
        weights.append(max_weight - line[1] + 1)
    while len(new_gen) < len(gen):  # generating childs

        parent = get_rand_parents(remaining_gen, max_weight, weights)
        # print("I am here")
        # print(parent[0])
        # print("its over")
        child1, child2, fit_1, fit_2 = generate_child(parent[0], parent[1])

        if (fit_1 == True):
            new_gen.append(child1)

        if (fit_2 == True):
            new_gen.append(child2)

    new_gen = sorted(new_gen, key=lambda x: x[1])

    while len(new_gen) > 100:
        # print("stuck")
        new_gen = new_gen[:-1]

    return new_gen





def random_pattern(queens):
    start_pattern = np.zeros((n, n), dtype=np.int64)
    for i in range(0, n):  # put weighted queens randomly in each column
        r = random.randint(0, n - 1)
        start_pattern[r][i] = queens[i]

    for i in range(0, q):
        while True:
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if start_pattern[a][b] != 0:
                continue
            start_pattern[a][b] = queens[n + i]
            break
    start_pattern = start_pattern.tolist()
    return start_pattern


def readCSV(file_address = ""):

    # default file address
    if len(file_address) == 0:
        # file_address = "F:\ML and CV\WPI_AI\Assignment_1\\board"
        file_address = "E:\Study\Sem 2\Artificial Intelligence\Project 1\Board.csv"

    board = []

    with open(file_address, mode='r',encoding='utf-8-sig')as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            board.append(list(lines))

        for rows in range(len(board)):
            print(board[rows])

    file.close()
    return board


if __name__ == '__main__':

    # select your way of input
    print(
        "Please select a method out of below options to generate nxn heavy queens grid\n1. Generate a random nxn grid\n2. Input a csv file\nType 1 or 2\n")
    method = input()
    if method == '1':
        print("Enter the grid size, n : ")
        n = input()  # input the grid size
        n = int(n)
        q = int(n / 8)
        total_queens = n + q
        queens = []  # make an array of wighted queens
        for i in range(0, n + q):
            r = random.randint(1, 9)  # generate a random no. between 0 and 9
            queens.append(r)

        b = []  # list to keep random generated column for extra queens
        for i in range(0, q):
            b1 = random.randint(0, n - 1)
            b.append(b1)

        start_pattern = random_pattern(queens)
        for line in start_pattern:
            print('  '.join(map(str, line)))

    elif method == '2':
        print("Please enter the address of your csv file : ")
        address = input()
        start_pattern = readCSV(file_address=address)  # Read the csv file
        n = len(start_pattern)  # calculate the grid length value
        queens = []  # list of queen weights in respective columns
        b = []  # list to keep random generated column for extra queens
        for i in range(0, len(start_pattern)):
            count = 0
            for j in range(0, len(start_pattern)):
                start_pattern[j][i] = int(start_pattern[j][i])
                if start_pattern[j][i] != 0:
                    queens.append(start_pattern[j][i])
                    count += 1
            if count > 1:
                b.append(i)
        total_queens = len(queens)
        q = total_queens - n
    else:
        print("Invalid entry")

    # n = 8  # input the grid size
    # q = int(n / 8)
    # total_queens = n + q
    # queens = []  # make an array of wighted queens
    # for i in range(0, n + q):
    #     r = random.randint(1, 9)  # generate a random no. between 0 and 9
    #     queens.append(r)  # add that random no. in weighted queens array
    #
    #     # make a matrix of nxn grid(all elements initialised to zero)
    #
    # start_pattern = random_pattern(queens)1


    gen = first_gen(start_pattern, n=100)
    gen = sorted(gen, key=lambda x: x[1])
    # gen = np.array(gen)
    print(gen[0][0])
    print("-------------------")
    gen_number = 0

    t_init = time.time()
    t_end = copy.deepcopy(t_init)

    time_to_run = 20  # edit time\

    fittest_organism = []
    t = []
    total_fittest = []
    fittest_organism.append(gen[0][1])  # appending the most fit organism in gen 1

    while t_end - t_init < time_to_run:

        gen = get_new_gen(gen)
        print("population ", len(gen))
        print(gen[0][1])
        total_fittest.append(gen[0][1])
        # gen_number +=1
        t_end = time.time()
        print("time", t_end - t_init)
        t.append(t_end - t_init)

        if int((t_end - t_init) / 10) > len(fittest_organism):
            fittest_organism.append(gen[0][1])

    print("--------------------")
    print(gen[0][0])
    print("fittest")

    print(fittest_organism)
    print(total_fittest)
    print(time)
    # print(fittest_organism[:][1])
    # for line in gen:
    #     # print ('  '.join(map(str, line[0])))
    #     print(line[1])
