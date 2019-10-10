'''
CPSC 415 -- Homework #4 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import logging
import random
from wumpus import *
from collections import namedtuple

Result = namedtuple('Result','score num_steps')

class Suite():

    def __init__(self, seeds=range(10)):
        self.seeds = seeds

    def run(self, Explorer_class, max_steps=10000):
        scores = []
        for seed in self.seeds:
            logging.critical('Running seed {}...'.format(seed))
            random.seed(seed)
            we = WumpusEnvironment()
            we.add_thing(Explorer_class(), we.START_SQUARE)
            step = 0
            while step < max_steps  and  not we.should_shutdown():
                we.step()
                step += 1
            scores.append(Result(we.agents[0].performance, step))
        return scores
