import numpy as np

# Define the Q-learning function
def q_learning(maze, episodes):
    # Initialize Q-values and rewards
    q_values = np.zeros((len(maze), len(maze[0])))
    rewards = [[-1,-1,-1,-1], [-1,-1,-1,-1], [-1,-1,-1,-1], [-1,-1,-1,-1]]
    rewards[3][3] = 100
    # Set the learning rate and discount factor
    alpha = 0.8
    gamma = 0.95

    for episode in range(episodes):
        # Initialize starting position and step counter
        current_pos = [0, 0]
        steps = 0
        while current_pos != [3, 3]:
            steps += 1
            # Select the next move
            next_pos = [current_pos[0] + np.random.choice([-1, 0, 1]), current_pos[1] + np.random.choice([-1, 0, 1])]
            if next_pos[0] < 0 or next_pos[0] > 3 or next_pos[1] < 0 or next_pos[1] > 3:
                continue
            if maze[next_pos[0]][next_pos[1]] == 1:
                continue
            # Update the Q-value for the current position
            q_values[current_pos[0]][current_pos[1]] += alpha * (rewards[next_pos[0]][next_pos[1]] +
                                                               gamma * np.max(q_values[next_pos[0], next_pos[1]]) -
                                                               q_values[current_pos[0]][current_pos[1]])
            current_pos = next_pos
    return q_values

# Define the maze
maze = [[0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]]

# Run Q-learning algorithm
q_values = q_learning(maze, 1000)
print(q_values)

"""
The result :

[[19.71141157 28.24876089 27.98971205 30.00472121]
 [ 4.56202823  0.          0.         57.74111432]
 [-3.12494445  0.          0.         92.55598594]
 [ 3.52768312 -3.53521697 18.28058332  0.        ]]
"""
