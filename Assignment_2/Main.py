from statistics import mean
import numpy as np
import csv
import os
from math import inf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from numpy import sqrt
from sklearn.metrics import r2_score
import pickle

class CSV_:

    def __init__(self, path):
        self.CSVfilePath = path

    def readCSV_board(self, file_address=''):

        # default file address
        if len(file_address) == 0:
            file_address = self.CSVfilePath

        board_list = []
        cost_list = []

        with open(file_address, mode='r')as file:
            csvFile = csv.reader(file)
            board_size = int(next(csvFile)[0])  # first line is the length of board
            index = 0
            for lines in csvFile:

                # only when there are blank lines
                if index == 0:
                    board = []  # new board will start after cost
                    # continue

                if index < board_size:
                    board.append(list(lines))
                    index += 1
                    continue

                if index == board_size:
                    for i in range(0, len(board)):
                        for j in range(0, len(board)):    
                            board[j][i] = int(board[j][i])
                    board_list.append(board)
                    cost_list.append(int(lines[0]))
                    index = 0
                    # print(board)
                    
                    continue

        file.close()
        return board_list, cost_list

csv_ = CSV_("F:\Study\Artificial Intelligence - CS 534\Assignments\Assignment_2\\Data\\N8.txt")
board_list, cost_list = csv_.readCSV_board("F:\Study\Artificial Intelligence - CS 534\Assignments\Assignment_2\\Data\\N8.txt")
# print(cost_list[-1])
# print(len(cost_list))
# for i in range(0,10000):
# for line in board_list[0]:
#     print ('  '.join(map(str, line)))
# print("\n")
# # print(board_list[])
# print(len(board_list))

n= len(board_list[0])

def attacking_queens(grid):
        totalhcost = 0
        totaldcost = 0
        for i in range(0,n):
            for j in range(0,n):
                #if this node is a queen, calculate all violations
                if grid[i][j] !=0:
                #subtract 2 so don't count self
                #sideways and vertical
                    totalhcost -= 2
                    for k in range(0,n):
                        if grid[i][k] !=0:
                            totalhcost += 1
                        if grid[k][j] !=0:
                            totalhcost += 1
                  #calculate diagonal violations
                    k, l = i+1, j+1
                    while k < n and l < n:
                        if grid[k][l] !=0:
                            totaldcost += 1
                        k +=1
                        l +=1
                    k, l = i+1, j-1
                    while k < n and l >= 0:
                        if grid[k][l] !=0:
                            totaldcost += 1
                        k +=1
                        l -=1
                    k, l = i-1, j+1
                    while k >= 0 and l < n:
                        if grid[k][l] !=0:
                            totaldcost += 1
                        k -=1
                        l +=1
                    k, l = i-1, j-1
                    while k >= 0 and l >= 0:
                        if grid[k][l] !=0:
                            totaldcost += 1
                        k -=1
                        l -=1
        return ((totaldcost + totalhcost)/2)

def queen_positions(grid):
    queen_pos = []
    for i in range(0,len(grid)):
        for j in range(0,len(grid)):
            if grid[j][i] !=0:
                queen_pos.append(j)
                continue
    return queen_pos

def queen_weights(grid):
    queen_weight = []
    for i in range(0,len(grid)):
        for j in range(0,len(grid)):
            if grid[j][i] !=0:
                queen_weight.append(grid[j][i])
                continue
    return queen_weight

def heaviest_Q(grid):
    Qboard = queen_weights(grid)
    HeavyQ = max(Qboard)
    return HeavyQ

def avg_weight(grid):
    Qboard = queen_weights(grid)
    avg = mean(Qboard)
    return avg

class training_sample:

#define attributes of the training samples we need i.e. initial pattern, solved pattern and solved pattern cost by astar search algorithm.

    def __init__(self, pattern, cost):
        self.pattern = pattern
        self.cost = cost
        self.attacking_pairs = attacking_queens(self.pattern)        # feature 1
        self.heaviest_queen = heaviest_Q(self.pattern)               # feature 2
        self.average_weight = avg_weight(self.pattern)               # feature 3
        # self.individual_attacks = 
        # self.

m = 3 #no. of features
X =  np.empty(shape=(1,m))            # features matrix for ML model
Y = np.reshape(cost_list, (len(cost_list), 1))             # target matrix for ML model
sample_node_list = []                   # nodes of training samples

for i in range(0,len(board_list)):
    current = training_sample(board_list[i], cost_list[i])
    sample_node_list.append(current)
    new_row = [current.attacking_pairs, current.heaviest_queen, current.average_weight]
    X = np.vstack([X, new_row])
X = X[1:,:]
Xnew = np.hstack([X,Y])

def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))

# model = LinearRegression()
# inputs = X
# targets = Y
# model.fit(inputs, targets)
# predictions = model.predict(inputs)
# loss = rmse(targets, predictions)
# print(loss)

data = pd.DataFrame(Xnew, columns = ['attacks','heavy','avg','cost'])
X_inputs = data[['attacks','heavy','avg']]
Y_targets = data['cost']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_inputs, Y_targets, test_size=0.1)

def train_and_evaluate(X_train, train_targets, X_val, val_targets):
    model = LinearRegression()
    model.fit(X_train, train_targets)
    train_rmse = rmse(model.predict(X_train), train_targets)
    val_rmse = rmse(model.predict(X_val), val_targets)
    return model, train_rmse, val_rmse

kfold = KFold(n_splits=10)
models = []

for train_idxs, val_idxs in kfold.split(Xtrain):
    X_train = Xtrain.iloc[train_idxs] 
    train_targets =Ytrain.iloc[train_idxs]
    X_val, val_targets = Xtrain.iloc[val_idxs], Ytrain.iloc[val_idxs]
    model, train_rmse, val_rmse = train_and_evaluate(X_train, 
                                                     train_targets, 
                                                     X_val, 
                                                     val_targets, 
                                                     )
    models.append(model)
    print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmse, val_rmse))

def predict_avg(models, inputs):
    return np.mean([model.predict(inputs) for model in models], axis=0)

preds = predict_avg(models, Xtest)

loss = rmse(Ytest, preds)
print("Error",loss)
accuracy = r2_score(Ytest, preds)
print("Accuracy",accuracy)
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


        

