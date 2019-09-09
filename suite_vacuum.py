'''
CPSC 415 -- Homework #1 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import importlib
import subprocess
import math
import os
import sys
from functools import reduce
import operator
import itertools
import logging
import random
from collections import namedtuple
import pandas as pd
import statistics

from vacuum import *

Result = namedtuple('Result','score num_steps')

class Suite():

    NUM_CORES = 8

    def __init__(self, base_seed=0, num_seeds=10):
        self.base_seed = base_seed
        self.num_seeds = num_seeds

    def run(self, userid, max_steps=10000):
        try:
            stud_module = importlib.import_module(userid + '_vacuum')
            vacuum_class = getattr(stud_module,userid.capitalize() +
                'VacuumAgent')
        except Exception as err:
            print(str(err))
            sys.exit(2)
        num_runs_per_core = math.ceil(self.num_seeds / self.NUM_CORES)
        starting_seeds_for_core = [ self.base_seed + i*num_runs_per_core 
            for i in range(self.NUM_CORES) ]
        procs = []
        output_files = []
        for start_seed in starting_seeds_for_core:
            ending_seed = min(self.base_seed + self.num_seeds - 1, 
                start_seed + num_runs_per_core - 1)
            logging.info('Running seeds {}-{}...'.format(start_seed,
                ending_seed))
            output_file = 'output{}.csv'.format(start_seed)
            cmd_line = ['./chunk_vacuum.py', userid, '{}-{}'.format(
                start_seed, ending_seed), str(max_steps) ]
            procs.append(subprocess.Popen(cmd_line))
            output_files.append(output_file)

        print('Waiting for completion...')
        [ p.wait() for p in procs ]
        print('...done.')

        with open('output.csv','w') as f:
            subprocess.Popen(['cat'] + output_files, stdout=f)

        os.system('grep -v "seed" output.csv > output2.csv')
        os.system('head -1 output.csv > outputheader.csv')
        os.system('cat outputheader.csv output2.csv > output3.csv')
        results = pd.read_csv('output3.csv')
        scores = results['score']
        steps = results['num_steps']
        with open('outputheader.csv','w') as f:
            print('# Score: min {}, max {}, med {}'.format(min(scores),
                max(scores), int(statistics.median(scores))), file=f)
            print('# Num steps: min {}, max {}, med {}'.format(min(steps),
                max(steps), int(statistics.median(steps))), file=f)
        os.system('cat outputheader.csv output3.csv > ' +
            'output_' + userid + '.csv')
        os.system('rm output[0-9]*.csv')
        os.system('head -2 output_' + userid + '.csv')
        print('Output in output_' + userid + '.csv.')
        med = statistics.median(scores)
        if med > 4000:
            print("\n{} earned +40XP! (the max)".format(userid))
        elif med < 0:
            print("\n{} earned +0XP, sorry!".format(userid))
        else:
            print("\n{} earned +{}XP!".format(userid,math.ceil(med / 100)))
