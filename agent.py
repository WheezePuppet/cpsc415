'''
CPSC 415
Stephen Davies, University of Mary Washington, fall 2019
Based somewhat on AIMA's git@github.com:aimacode/aima-python.git agents.py.
'''

import logging
import abc


class Thing(abc.ABC):
    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def is_alive(self):
        """Things that are 'alive' should return true."""
        return hasattr(self, 'alive') and self.alive


class Agent(Thing): 
    def __init__(self):
        self.alive = True
        self.performance = 0

    @abc.abstractmethod
    def program(self, percept):
        pass


