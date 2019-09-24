'''
CPSC 415 -- Homework #3 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import json
from pathlib import Path
import re
import random
import sys

CONFIG_DIR = 'chess_configs'

class Config():

    def __init__(self, config_file_basename, crazy=False):
        filename = CONFIG_DIR + '/' + config_file_basename
        if '.config' not in filename:
            filename = filename + '.config'
        if not Path(filename).is_file():
            print("Unknown config file {}. Valid config files are: {}".format(
                config_file_basename, ', '.join(Config.get_config_names())))
            sys.exit(2)
        with open(filename) as f:
            self.__dict__ = json.load(f)
        self.config_file_basename = config_file_basename
        import std_config
        for var in [ var for var in dir(std_config)
                                            if not var.startswith('__') ]:
            self.__dict__[var] = getattr(std_config, var)
        self.__dict__['CRAZY'] = crazy
        if crazy:
            self.crazify()

    def crazify(self):
        for col in self.X_AXIS_LABELS:
            piece = random.choice(list(
                set(self.SHORT_NAME.keys()) - {'K','P'}))
            self.START_POSITION[col + '1'] = piece.upper()
            self.START_POSITION[col + str(self.NUM_ROWS)] = piece.lower()
        king_col = random.choice(self.X_AXIS_LABELS)
        self.START_POSITION[king_col + '1'] = 'K'
        self.START_POSITION[king_col + str(self.NUM_ROWS)] = 'k'
        

    @staticmethod
    def get_config_names():
        # Auto-retrieving config files not working on Windows for some reason.
        return ['reg','mini','large']
