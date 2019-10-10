'''
CPSC 415 -- Homework #4 support file
Stephen Davies, University of Mary Washington, fall 2017
'''

from wumpus import ExplorerAgent

import random

class manual_ExplorerAgent(ExplorerAgent):
    '''Note: run this in "auto" mode.'''

    def __init__(self):
        super().__init__()

    def program(self, percept):
        print("The percept I just got is: {}".format(percept))
        action = input(', '.join([str(i) + '.' + x for i, x in enumerate(
                self.possible_actions)]) + '? ')
        return self.possible_actions[int(action)]
