#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from kaggle_environments import make, evaluate, utils
import inspect
import sys


# In[2]:


def agent_minimax_play(obs, config):
    import numpy as np    
    import random
    
    NUM_STEPS_LOOKAHEAD = 3
    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)
    
    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # 1.) Checking horizontal orientation
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # 2.) Checking vertical orientation
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # 3.) Checking positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # 4.) Checking negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows
    
    # Helper function for minimax: calculates value of heuristic for grid
    def get_heuristic(grid, mark, config):
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        num_fours_opp = count_windows(grid, 4, mark%2+1, config)
        score = 1*num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours
        return score
    
    def get_minimax_score_for_move(grid, col, your_mark, alphabeta, config, nsteps):

        # Play your move as the maximizingPlayer
        next_grid = drop_piece(grid, col, your_mark, config)
        if alphabeta:        
            minimax_score = minimax_alphabeta(node=next_grid, depth=nsteps-1, maximizingPlayer=False, alpha=-np.Inf, beta=np.Inf, your_mark=your_mark, config=config)
        else:
            minimax_score = minimax(node=next_grid, depth=nsteps-1, maximizingPlayer=False, your_mark=your_mark, config=config)
        # Since you have already played your move due to the drop_piece method, so
        # depth = nsteps-1 so we traversed 1 depth
        # maximizingPlayer argument is False i.e. indicating a minimizingPlayer 
        # maximizingPlayer = you -> have already played your move
        return minimax_score
    
    def drop_piece(grid, col, mark, config):
        new_grid = grid.copy()
        for r in range(config.rows-1,-1,-1):
            if new_grid[r,col] == 0:
                new_grid[r,col] = mark
                return new_grid            
    
    def is_terminal_window(window, config):
        if window.count(1)==config.inarow or window.count(2)==config.inarow:
            return True
    
    def is_terminal_node(grid, config):
        # How can you term a grid as a terminal node i.e. beyond which the game is not possible
        # Scenario#1: no further move is possible

        if sum(grid[0,:]==0)==0:
            return True

        # Scenario#2: opponent already got a config.inarow number
        # Now lets check all possible orientations:
        # i.e. 1.) horizontal 2.) vertical 3.) positive diagonal 4.) negative diagonal

        # For 1.) horizontal
        for row in range(config.rows):
            for col in range((config.rows-config.inarow)+1):
                window = list(grid[row,range(col,col+config.inarow)])
                if is_terminal_window(window, config):
                    return True

        # For 2.) vertical
        for row in range((config.rows-config.inarow)+1):
            for col in range(config.columns):
                window = list(grid[range(row,row+config.inarow),col])
                if is_terminal_window(window, config):
                    return True

        # For 3.) +ve diagonal
        for row in range((config.rows-config.inarow)+1):
            for col in range((config.rows-config.inarow)+1):
                window = list(grid[range(row,row+config.inarow),range(col,col+config.inarow)])
                if is_terminal_window(window, config):
                    return True

        # For 4.) -ve diagonal
        for row in range(config.inarow-1,config.rows):
            for col in range((config.rows-config.inarow)+1):
                window = list(grid[range(row,row-config.inarow,-1),range(col,col+config.inarow)])
                if is_terminal_window(window, config):
                    return True

        return False
    
    def minimax_alphabeta(node, depth, maximizingPlayer, alpha, beta, your_mark, config):
    
        list_available_moves = [col for col in range(config.columns) if node[0,col]==0]

        # 3 scenarios to handle
        # Scenario 1: reached the end i.e. 
        # Condition A - no further to traverse, or
        # Condition B - its a terminal node i.e. no further available moves, game over opponent won

        if depth==0 or is_terminal_node(node, config):
            return get_heuristic(node,your_mark,config)
        

        if maximizingPlayer:
            value = -np.Inf        
            for col in list_available_moves:
                child = drop_piece(node, col, your_mark, config)
                value = max(value, minimax_alphabeta(child, depth-1, False, alpha, beta, your_mark, config))
                alpha = max(alpha, value)
                if alpha > beta:
                    break
            return value


        else:
            value = np.Inf
            for col in list_available_moves:
                child = drop_piece(node, col, your_mark%2+1, config)
                value = min(value, minimax_alphabeta(child, depth-1, True, alpha, beta, your_mark, config))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value
    
    def minimax(node, depth, maximizingPlayer, your_mark, config):
    
        list_available_moves = [col for col in range(config.columns) if node[0,col]==0]

        # 3 scenarios to handle
        # Scenario 1: reached the end i.e. 
        # Condition A - no further to traverse, or
        # Condition B - its a terminal node i.e. no further available moves, game over opponent won

        if depth==0 or is_terminal_node(node, config):
            return get_heuristic(node,your_mark,config)


        if maximizingPlayer:
            value = -np.Inf        
            for col in list_available_moves:
                child = drop_piece(node, col, your_mark, config)
                value = max(value, minimax(child, depth-1, False, your_mark, config))
            return value


        else:
            value = np.Inf
            for col in list_available_moves:
                child = drop_piece(node, col, your_mark%2+1, config)
                value = min(value, minimax(child, depth-1, True, your_mark, config))
            return value
    
    
    # Step1. Convert the board list to a grid
    
    board_array = np.array(obs.board).reshape(config.rows,config.columns)
    
    
    # Step2. Get list of allowed moves
    # How can you get a list of allowed moves ? Note a move is valid if there is any empty row in a column
    
    list_allowed_moves = [c for c in range(config.columns) if (sum(board_array[:,c]==0)>0)]
    
    # or later
    # for first turn -
    # I am planning to replace it by configuring to:
    # A. if turn = first: middle move
    # B. if turn is not first: either left or right of middle
    
    
    
    # Step3. Now for each of the move within the list_allowed_moves, lets generate a heuristic score using minimax
    alphabetamode = True
    move_score_dict = {}
    for allowed_move in list_allowed_moves:
        # obs.mark - the peice assigned to the agent (either 1 or 2)
        minimax_score = get_minimax_score_for_move(board_array, allowed_move, obs.mark, alphabetamode, config, NUM_STEPS_LOOKAHEAD)
        move_score_dict[allowed_move] = minimax_score
    
    # Step4. Trying to obtain the list of allowed moves for which the score is the highest
    
    max_score = -np.inf
    
    # Finding max score
    for move,score in move_score_dict.items():
        if score > max_score:
            max_score = score
    
    moves_with_max_score = []
    for move,score in move_score_dict.items():
        if score >= max_score:
            moves_with_max_score.append(move)
            
    # Step5. Now as a final step returning the move
    play_move = random.choice(moves_with_max_score)
    
    return play_move


# In[3]:


# Create the game environment
env = make("connectx")

# Two random agents play one game round
env.run([agent_minimax_play, "random"])

# Show the game
env.render(mode="ipython")


# In[4]:


def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(agent_minimax_play, "submission.py")


# In[5]:


out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


# In[ ]:




