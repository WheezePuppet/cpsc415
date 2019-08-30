#!/usr/bin/env python3
'''
CPSC 415 -- Homework #1 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import logging
import importlib
import sys

from environment import *
from agent import *
from vacuum import *
from visualize import *
from suite_vacuum import *

''' 
To run this program, make sure you have a Python file in the current directory
whose name is exactly your lower-case UMW Net ID (e.g., "jsmith19") followed
by the string "_vacuum.py". In that file, make sure you have a class defined
whose name is exactly your first-letter-capitalized UMW Net ID followed by the
string "VacuumAgent" that inherits from Agent in agent.py. Then, run it via:

$ python3 main_vacuum.py jsmith19

You can have it run continuously until the whole floor is clean by doing:

$ python3 main_vacuum.py jsmith19 auto

You can (re-)use a specific seed by doing:

$ python3 main_vacuum.py jsmith19 auto 293

You can run a whole (non-visual) suite of many (say, 47) simulations this way:

$ python3 main_vacuum.py jsmith19 suite=47

and finally, you can set the debug level by doing this sort of thing:

$ python3 main_vacuum.py jsmith19 auto 0 DEBUG
    or
$ python3 main_vacuum.py jsmith19 interactive 4449 INFO

(A seed of 0, or an omitted seed, will give a fresh seed each time.)
'''

def print_usage():
    print('Usage: main_vacuum.py UMWNetID [interactive|auto|suite=#] ' +
        '[seed] [debugLevel].')

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
    logging.getLogger().setLevel(sys.argv[4])
else:
    logging.getLogger().setLevel('CRITICAL')

try:
    stud_module = importlib.import_module(sys.argv[1] + '_vacuum')
    Stud_agent_class = getattr(stud_module,sys.argv[1].capitalize() +
        'VacuumAgent')
except Exception as err:
    print(str(err))
    sys.exit(2)

if not issubclass(Stud_agent_class, VacuumAgent):
    print("{} doesn't inherit from VacuumAgent.".format(
        Stud_agent_class.__name__))
    sys.exit(3)

# Go!
if do_suite:
    results = Suite(seed,num_runs).run(sys.argv[1], 1000)
else:
    print('Using seed {}.'.format(seed))
    random.seed(seed)
    dve = RandomDirtyVacuumEnvironment()
    dve.add_to_random_empty_square(Stud_agent_class())
    ve = VisualXYEnvironment(dve, 50, 50, 'Vacuum cleaner world')
    ve.start(interactive)
