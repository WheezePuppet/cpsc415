#!/usr/bin/env python3
'''
CPSC 415 -- Homework #4 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import logging
import importlib
import sys
import statistics

from environment import *
from agent import *
from wumpus import *
from visualize import *


def print_usage():
    print('Usage: main_wumpus.py explorer_name [interactive|auto|suite=#] ' +
        '[seed] [debugLevel|NONE].')

if len(sys.argv) not in [2,3,4,5]:
    print_usage()
    sys.exit(1)

if len(sys.argv) > 2:
    interactive = sys.argv[2] == 'interactive'
    do_suite = sys.argv[2].startswith('suite=')
    if do_suite:
        try:
            num_runs = int(sys.argv[2][sys.argv[2].index('=')+1:])
        except:
            print_usage()
            print("('suite=#' parameter must be numeric.)")
            sys.exit(2)
else:
    interactive = True
    do_suite = False

if len(sys.argv) > 3:
    try:
        seed = int(sys.argv[3])
        if seed == 0:
            seed = random.randrange(10000)
    except:
        print_usage()
        print("('seed' must be numeric.)")
        sys.exit(3)
else:
    seed = random.randrange(10000)

if len(sys.argv) > 4:
    if sys.argv[4] == 'NONE':
        logging.getLogger().setLevel(logging.CRITICAL + 1)
    else:
        logging.getLogger().setLevel(sys.argv[4])
else:
    logging.getLogger().setLevel('CRITICAL')

try:
    stud_module = importlib.import_module(sys.argv[1] + '_ExplorerAgent')
    Explorer_class = getattr(stud_module,sys.argv[1] + '_ExplorerAgent')
except Exception as err:
    print(str(err))
    sys.exit(2)

if not issubclass(Explorer_class, ExplorerAgent):
    print("{} doesn't inherit from ExplorerAgent.".format(
        Explorer_class.__name__))
    sys.exit(3)

# Go!
if do_suite:
    from suite_wumpus import *
    results = Suite(range(seed,seed+num_runs)).run(Explorer_class)
    scores, steps = ([ score for score,_ in results ],
                    [ steps for _,steps in results ])
    print('seed,score,num_steps')
    for i,(score, num_steps) in enumerate(results):
        print('{},{},{}'.format(seed+i,score,num_steps))
    print('# Score: min {}, max {}, med {}'.format(min(scores),max(scores),
        int(statistics.median(scores))))
    print('# Num steps: min {}, max {}, med {}'.format(min(steps),max(steps),
        int(statistics.median(steps))))
else:
    # Actually instantiate the student's agent.
    explorer = Explorer_class()
    print('Using seed {}.'.format(seed))
    random.seed(seed)
    we = WumpusEnvironment()
    we.add_thing(explorer, we.START_SQUARE)
    ve = VisualXYEnvironment(we, 100, 100, 'Wumpus world')
    ve.start(interactive)
