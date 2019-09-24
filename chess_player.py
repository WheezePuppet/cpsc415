'''
CPSC 415 -- Homework #3 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import abc
import logging
import random
import operator
import inspect
import os
from copy import deepcopy


class ChessPlayer(abc.ABC):

    def __init__(self, board, color):
        self.board = board
        self.color = color

    @abc.abstractmethod
    def get_move(self, your_remaining_time, opp_remaining_time, prog_stuff):
        pass

    @staticmethod
    def get_player_names():
        names = []
        filenames = [ f for f in os.listdir('.') if os.path.isfile(f) and
            f.endswith('_ChessPlayer.py') ]
        for filename in filenames:
            mod_name = filename.replace('.py','')
            names.append(mod_name)
        return names

