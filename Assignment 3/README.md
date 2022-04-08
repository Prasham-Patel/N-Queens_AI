# CS 534 - Artificial Intelligence | Assignment-3 | Group-5 | README file

## Input arguments

  There are total six input arguments:
  - **Filename**: address of the file containig grid 
  - **Reward**: reward given to agent per movment
  - **Gamma**: the discount factor for learning 
  - **Time to learn**: time in seconds given to agent for learning  
  - **Movement probability**: the propability of agent to take the desired action 
  - **Policy**: an interger from [0, 3]
      - 0 is random selection policy, each action has a probability of 0.25
      - 1 is the epsilon greedy policy with epsilon = 0.1
      - 2 is the counter update policy, discovers each grid box a minimum of 5 times
      - 3 is the proposed policy, exploration depends on the size of grid and time given to learn
  - **Learning Algorithm**: both SARSA and Q-learning are implemented:
      - enter "SARSA" to run SARSA
      - enter "Q_learning" to run Q-learning


