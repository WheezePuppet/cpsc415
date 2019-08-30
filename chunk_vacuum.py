#!/usr/bin/env python3
'''
CPSC 415 -- Homework #1 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import subprocess
import importlib
import os
import sys
from functools import reduce
import operator
import itertools
import logging
import random
from collections import namedtuple

from vacuum import *

Result = namedtuple('Result','seed score num_steps')

class Chunk():

    def __init__(self, seeds=range(10)):
        self.seeds = seeds

    def run(self, vacuum_class, max_steps=10000):
        scores = []

        for seed in seeds:
            logging.debug('Running seed {}...'.format(seed))
            random.seed(seed)
            dve = RandomDirtyVacuumEnvironment()
            dve.add_to_random_empty_square(vacuum_class())
            step = 0
            steps_to_clean = 0
            while step < max_steps:
                dve.step()
                step += 1
                if not dve.is_clean():
                    steps_to_clean += 1
            scores.append(Result(seed, dve.agents[0].performance,
                steps_to_clean))
        return scores


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: chunk_vacuum.py UMWNetID startSeed-endSeed [maxSteps].")
        sys.exit(1)
    if len(sys.argv) == 3:
        max_steps = 1000
    else:
        max_steps = int(sys.argv[3])
    start,end = [ int(s) for s in sys.argv[2].split('-') ]
    seeds = range(start,end+1)

    try:
        stud_module = importlib.import_module(sys.argv[1] + '_vacuum')
        Stud_agent_class = getattr(stud_module,sys.argv[1].capitalize() +
            'VacuumAgent')
    except Exception as err:
        print(str(err))
        sys.exit(2)

    chunk = Chunk(seeds)
    scores = chunk.run(Stud_agent_class, max_steps)
    output_file = 'output'+str(start)+'.csv'
    with open(output_file, 'w') as f:
        print('seed,score,num_steps',file=f)
        for score in scores:
            print("{},{},{}".format(score.seed,score.score,score.num_steps),
                file=f)
