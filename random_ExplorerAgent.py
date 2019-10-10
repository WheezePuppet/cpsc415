'''
CPSC 415 -- Homework #4 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

from wumpus import ExplorerAgent

import random
import logging

class random_ExplorerAgent(ExplorerAgent):

    def __init__(self):
        super().__init__()

    def program(self, percept):
        action = random.choice(self.possible_actions)
        logging.debug("Doing action {}.".format(action))
        return action
