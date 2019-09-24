#!/usr/bin/env python3
'''
CPSC 415 -- Homework #3 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import sys
import os
import glob
import json
import subprocess

def print_usage():
    print('Usage: chess_match.py [num_games=6]\n' +
        '                      [config_file=configFileName]\n' +
        '                      [crazy=True/False]\n' +
        '                      agent1=computerPlayer1UMWNetId\n' +
        '                      agent2=computerPlayer2UMWNetId.')

if any(['usage' in x for x in sys.argv]):
    print_usage()
    sys.exit(1)

this_module = sys.modules[__name__]

params = [
    ('num_games',6),
    ('config_file','mini'),
    ('crazy',True),
    ('agent1',None),
    ('agent2',None) ]

for arg in sys.argv[1:]:
    if not '=' in arg:
        print("Malformed argument '{}'.".format(arg))
        print_usage()
        sys.exit(2)
    arg_name, arg_val = arg.split('=')
    if arg_name not in [ x for x,_ in params ]:
        print("Unknown argument '{}'.".format(arg_name))
        print_usage()
        sys.exit(3)
    setattr(this_module, arg_name, arg_val)

for (param,default) in params:
    if not hasattr(this_module, param):
        setattr(this_module, param, default)

if agent1 and agent2:
    from chess_headless import HeadlessGame
    num_games = int(num_games)
    print("Starting {}-game match between {} and {}.".format(num_games,
        agent1,agent2))
    procs = []
    for game_num in range(num_games):
        procs.append(subprocess.Popen([ './main_chess.py',
            'config_file={}'.format(config_file),
            'crazy={}'.format(crazy),
            'agent1={}'.format(agent1),
            'agent2={}'.format(agent2),
            'log_file_suffix={}'.format(game_num) ]))
        agent1,agent2 = agent2,agent1
    print('Waiting for completion...')
    [ p.wait() for p in procs ]
    print('...done.')
    num_agent1_wins = int(subprocess.check_output(
        ("grep -h WINNER {agent1}_vs_{agent2}*.log {agent2}_vs_{agent1}*.log"+
         "| grep {agent1} | wc -l").format(
            agent1=agent1,agent2=agent2), shell=True))
    num_agent2_wins = int(subprocess.check_output(
        ("grep -h WINNER {agent2}_vs_{agent1}*.log {agent1}_vs_{agent2}*.log"+
         "| grep {agent2} | wc -l").format(
            agent1=agent1,agent2=agent2), shell=True))
    num_draws = int(subprocess.check_output(
        ("grep -h WINNER {agent2}_vs_{agent1}*.log {agent1}_vs_{agent2}*.log"+
         "| grep draw | wc -l").format(
            agent1=agent1,agent2=agent2), shell=True))
    logs = []
    game_log_files = glob.glob("{a1}_vs_{a2}*.log".format(a1=agent1,a2=agent2))
    game_log_files.extend(glob.glob("{a2}_vs_{a1}*.log".format(a1=agent1,
        a2=agent2)))
    for f in game_log_files:
        print("opening {}...".format(f))
        with open(f) as fp:
            logs.append(json.load(fp))
    match_log = {}
    match_log['OPPONENT1'] = agent1
    match_log['OPPONENT2'] = agent2
    match_log['WINNER'] = agent1 if num_agent1_wins > num_agent2_wins \
        else agent2 if num_agent2_wins > num_agent1_wins else 'draw'
    match_log['NUM_GAMES'] = num_games
    match_log[agent1 + '_WINS'] = num_agent1_wins
    match_log[agent2 + '_WINS'] = num_agent2_wins
    match_log['DRAWS'] = num_draws
    match_log['CONFIG'] = config_file
    match_log['CRAZY'] = crazy
    match_log['GAMES'] = logs
    with open("match{n}_{a1}_vs_{a2}.log".format(a1=agent1,a2=agent2,
        n=num_games),"w") as fp:
        json.dump(match_log,indent=4,fp=fp)
    [ os.remove(f) for f in game_log_files ]
    print("***************************************************************")
    print("{} won {} times ({:.1f}%).".format(agent1,num_agent1_wins,
        num_agent1_wins/num_games*100))
    print("{} won {} times ({:.1f}%).".format(agent2,num_agent2_wins,
        num_agent2_wins/num_games*100))
    print("There were {} draws ({:.1f}%).".format(
        num_draws,num_draws/num_games*100))
else:
    print_usage()
