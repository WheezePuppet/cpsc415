'''
CPSC 415 -- Homework #1 support file
Stephen Davies, University of Mary Washington, fall 2019
Based somewhat on AIMA's git@github.com:aimacode/aima-python.git agents.py.
'''

import logging
import sys
import random

from agent import *
from environment import *



class VacuumAgent(Agent):
    image_filename = 'cleaner.gif'
    possible_actions = ['Right', 'Left', 'Up', 'Down', 'Suck', 'NoOp']
    def __init__(self):
        super().__init__()
        self._bump = False


class VacuumEnvironment(XYEnvironment):

    """The environment of [Ex. 2.12]. Agent perceives dirty or clean, and bump
    (into obstacle) or not; 2D discrete world of unknown size; performance
    measure is 40 for each square of dirt cleaned, -5 for each 'suck' action,
    and -2 for each move action."""

    def __init__(self, width=10, height=10):
        super().__init__(width, height)
        self.add_walls()

    def thing_classes(self):
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,
                TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def percept(self, agent):
        """The percept is a tuple of ('Dirty' or 'Clean', 'Bump' or 'None').
        Unlike the TrivialVacuumEnvironment, location is NOT perceived."""
        status = ('Dirty' if self.some_things_at(
            self[agent], Dirt) else 'Clean')
        bump = ('Bump' if agent._bump else 'None')
        return (status, bump)

    def execute_action(self, agent, action):
        if action not in agent.possible_actions:
            print("Illegal action {}! Shutting down.".format(action))
            sys.exit(1)
        if action == 'Suck':
            agent.performance -= 5
            dirt_list = self.list_things_at(self[agent], Dirt)
            if dirt_list != []:
                dirt = dirt_list[0]
                agent.performance += 40
                self.delete_thing(dirt)
        else:
            super().execute_action(agent, action)

        if action in {'Left','Right','Up','Down'}:
            agent.performance -= 2

    def is_clean(self):
        return all([ type(thing) is not Dirt for thing in self ])

    def should_shutdown(self):
        return self.is_clean()

    def add_to_random_empty_square(self, thing):
        possible_squares = [(x,y) for x in range(self.width) 
            for y in range(self.height) if (x,y) not in self.values()]
        self.add_thing(thing,random.choice(possible_squares))
        

class DirtyVacuumEnvironment(VacuumEnvironment):

    def __init__(self, width=10, height=10, dirt_prob=.5):
        super().__init__(width, height)
        self.width=width
        self.height=height
        self.dirt_prob=dirt_prob
        self._scatter_dirt()

    def _scatter_object(self, cls, prob=.5):
        '''Randomly put down an object of the class passed on every empty 
        square, with probability passed.'''
        possible_squares = [(x,y) for x in range(self.width) 
            for y in range(self.height) if (x,y) not in self.values()]
        for sq in possible_squares:
            if random.random() < prob:
                self.add_thing(cls(), sq)

    def _scatter_dirt(self):
        self._scatter_object(Dirt, self.dirt_prob)


class RandomDirtyVacuumEnvironment(DirtyVacuumEnvironment):

    def __init__(self, width_range=(10,20), height_range=(10,20)):
        super().__init__(
            random.randint(*width_range), random.randint(*height_range))
        self._add_walls()

    def _add_walls(self):
        self._scatter_object(Wall, .2)



class Dirt(Thing):
    image_filename = 'dirt.gif'
