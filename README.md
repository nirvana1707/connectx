# ConnectX
## Introduction
 https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression

As part of this task, different playing agents (with move strategies) had to be developed and competed against different players


## Background
ConnectX is a game, where the main objective is to get a certain number of your checkers in a row horizontally, vertically, or diagonally on the game board before your opponent. When it's your turn, you “drop” one of your checkers into one of the columns at the top of the board. Then, let your opponent take their turn. This means each move may be trying to either win for you, or trying to stop your opponent from winning.

## Methodology

Here, I use minimax algorithm with alpha-beta pruning to look N-steps ahead of the opponent.
Pseudo-code:
![alt text](https://github.com/nirvana1707/connectx/blob/main/images/pseudo_code.PNG)
Source:https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode

## Results
The leaderboard is not based on how accurate your algorithm is but rather how well you’ve performed against other users. 
My score in Public LB is 799.5 
Game simulation screenshots:

![alt text](https://github.com/nirvana1707/connectx/blob/main/images/game_screenshot1.PNG)

![alt text](https://github.com/nirvana1707/connectx/blob/main/images/game_screenshot2.PNG)
