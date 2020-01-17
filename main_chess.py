#!/usr/bin/env python3
'''
CPSC 415 -- Homework #3 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

# Long long ago, I used chapter 4 of Chaudhary's "Tkinter GUI Blueprints" as a
# starting point for this. It has since evolved unrecognizeably.

import sys
from pathlib import Path
import logging
import builtins
import json

import chess_config

def print_usage():
    print('Usage: main_chess.py [debug_level=debugLevel]\n' +
        '                     [config_file=configFileName]\n' +
        '                     [crazy=True/False]\n' +
        '                     [log_file_suffix=""]\n' +
        '                     [replay=logFileName]\n' +
        '                     [replay_speed=500]\n' +
        '                     [agent1=computerPlayer1UMWNetId]\n' +
        '                     [agent2=computerPlayer2UMWNetId].')

if any(['usage' in x for x in sys.argv]):
    print_usage()
    sys.exit(1)

this_module = sys.modules[__name__]

params = [
    ('debug_level','INFO'),
    ('config_file','reg'),
    ('crazy',False),
    ('replay',None),
    ('replay_speed',500),
    ('log_file_suffix',""),
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


logging.getLogger().setLevel(debug_level)

builtins.cfg = chess_config.Config(config_file, crazy)

if replay:
    with open(replay,'r') as f:
        saved_game = json.load(fp=f)
    import chess_view
    import tkinter as tk
    root = tk.Tk()
    the_view = chess_view.View(root)
    if int(replay_speed) < 300:
        print("(replay_speed minimum = 300, sorry.)")
        replay_speed = 300
    the_view.replay(saved_game, replay_speed)
    root.mainloop()
elif agent1 and agent2:
    from chess_headless import HeadlessGame
    print("Headless face-off between {} and {}.".format(agent1,agent2))
    game = HeadlessGame(agent1, agent2, config_file, crazy_mode=crazy,
        log_file_suffix=log_file_suffix)
    game.start_game()
elif agent1 or agent2:
    print("Either both agents must be specified, or neither.")
    print_usage()
    sys.exit(4)
else:
    import chess_view
    import tkinter as tk
    root = tk.Tk()
    chess_view.View(root)
    root.mainloop()
