'''
CPSC 415 -- Homework #3 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

# Standard configuration variables.

IMAGE_DIR = 'images'
SQUARE_HEIGHT = 64
SQUARE_WIDTH = 64
BOARD_COLOR_DARK = 'cornflower blue'
BOARD_COLOR_LIGHT = 'khaki'

TIME_LIMIT = 60*60   # (Seconds.) Total time to move for each player.
MAX_MOVES = 100      # Maximum number of moves per player before forced draw.

ORTHOGONAL_DIRS = tuple(
    (x,y) for x in range(-1,2) for y in range(-1,2) if abs(x-y)==1)
DIAGONAL_DIRS = tuple((x,y) for x in [-1,1] for y in [-1,1])
KNIGHT_DIRS = tuple((x,y) for x in [-2,-1,1,2] for y in [-2,-1,1,2]
    if abs(x)+abs(y) == 3)
FOOL_DIRS = tuple((x,y) for x in [-2,0,2] for y in [-2,0,2]
    if not (x == y == 0))

