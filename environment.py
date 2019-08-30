'''
CPSC 415
Stephen Davies, University of Mary Washington, fall 2019
Based somewhat on AIMA's git@github.com:aimacode/aima-python.git agents.py.
'''

import logging
import abc
import random
import collections

from agent import *



class Environment(collections.UserDict):
    '''An Environment object itself is a dictionary from contained things to
    their locations.'''
    def __init__(self):
        super().__init__()
        self.agents = []
        self.observers = []

    @abc.abstractmethod
    def percept(self, agent):
        """Return the percept that the agent sees at this point."""

    @abc.abstractmethod
    def execute_action(self, agent, action):
        """Change the world to reflect this action."""

    def add_observer(self, observer):
        self.observers.append(observer)

    def default_location(self, thing):
        """Default location to place a new thing with unspecified location."""
        return None

    def exogenous_change(self):
        """If there is spontaneous change in the world, override this."""
        pass

    def is_done(self):
        """By default, we're done when we can't find a live agent."""
        return not any(agent.is_alive() for agent in self.agents)

    def step(self):
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do. If there are interactions between them, you'll need to
        override this method."""
        if not self.is_done():
            actions = []
            for agent in self.agents:
                if agent.alive:
                    actions.append(agent.program(self.percept(agent)))
                else:
                    actions.append("")
            for (agent, action) in zip(self.agents, actions):
                self.execute_action(agent, action)
            self.exogenous_change()

    def run(self, steps=1000):
        """Run the Environment for given number of time steps."""
        for step in range(steps):
            if self.is_done():
                return
            self.step()

    def list_things_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        return [thing for thing, loc in self.items()
                if loc == location and isinstance(thing, tclass)]

    def some_things_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_things_at(location, tclass) != []

    def add_thing(self, thing, location=None):
        """Add a thing to the environment, setting its location. For
        convenience, if thing is an agent program we make a new agent
        for it. (Shouldn't need to override this."""
        if not isinstance(thing, Thing):
            thing = Agent(thing)
        if thing in self:
            print("Can't add the same thing twice")
        else:
            self[thing] = (location if location is not None
                                        else self.default_location(thing))
            if isinstance(thing, Agent):
                thing.performance = 0
                self.agents.append(thing)

    def delete_thing(self, thing):
        """Remove a thing from the environment."""
        try:
            del self[thing]
        except ValueError as e:
            pass
        if thing in self.agents:
            self.agents.remove(thing)
        for obs in self.observers:
            obs.thing_deleted(thing)

    def move_to(self, thing, destination):
        '''Move the thing to a new location. (Being in the plain Environment
        class, this has nothing intrinsically to do with a coordinate system
        or the like.)'''
        self[thing] = destination
        self.notify_observers(thing)

    def notify_observers(self, thing):
        for o in self.observers:
            o.thing_moved(thing)

    def should_shutdown(self):
        '''Can be overridden by specialized environments to signal a stopping
        condition.'''
        return False


class XYEnvironment(Environment):
    """This class is for environments on a 2D plane, with locations
    labelled by (x, y) points, either discrete or continuous.
    """

    def __init__(self, width=10, height=10):
        super().__init__()

        self.width = width
        self.height = height

        # Sets iteration start and end (no walls).
        self.x_start, self.y_start = (0, 0)
        self.x_end, self.y_end = (self.width, self.height)

    perceptible_distance = 1

    def things_near(self, location, radius=None, manhattan=True):
        """Return all things within radius of location."""
        if radius is None:
            radius = self.perceptible_distance
        if manhattan:
            return [(thing, abs(loc[0] - location[0]) + 
                            abs(loc[1] - location[1]))
                for thing, loc in self.items() if 
                            abs(loc[0] - location[0]) + 
                            abs(loc[1] - location[1]) <= radius]
        radius2 = radius * radius
        def dist_sq(x,y):
            return x^2 + y^2
        return [(thing, radius2 - dist_sq(location, self[thing]))
                for thing, loc in self.things if dist_sq(location, loc) 
                                                                <= radius2]

    def percept(self, agent):
        """By default, agent perceives things within a default radius."""
        return self.things_near(self[agent])

    def execute_action(self, agent, action):
        '''Have the agent carry out this action. If a move in a compass
        direction, it may work, or may not, depending on whether there's an
        obstacle there. The next percept (2nd element of tuple) will tell the
        agent whether this happened.'''
        agent._bump = False
        if action in ['Left','Up','Right','Down']:
            agent._bump = self.try_to_move_in_dir(agent, action)
        elif action == 'Forward':
            agent._bump = self.try_to_move_in_dir(agent,
                agent._facing_direction)
        elif action == 'TurnLeft':
            directions = [ 'Up','Left','Down','Right','Up' ]
            agent._facing_direction = directions[
                directions.index(agent._facing_direction) + 1]
        elif action == 'TurnRight':
            directions = [ 'Up','Right','Down','Left','Up' ]
            agent._facing_direction = directions[
                directions.index(agent._facing_direction) + 1]
        elif action == 'NoOp':
            pass
        else:
            logging.critical("UNKNOWN action {}!!".format(action))
        self.notify_observers(agent)

    def try_to_move_in_dir(self, agent, direction):
        return self.move_to(agent, self.square_in_dir(direction, self[agent]))

    def square_in_dir(self, direction, location, num_steps=1):
        '''Return the location that is num_steps in the given direction from
        the given location.'''
        x,y = location
        if direction == 'Left':
            return (x-num_steps,y)
        elif direction == 'Right':
            return (x+num_steps,y)
        elif direction == 'Up':
            return (x,y+num_steps)
        elif direction == 'Down':
            return (x,y-num_steps)

    def default_location(self, thing):
        return (random.choice(self.width), random.choice(self.height))

    def move_to(self, thing, destination):
        thing._bump = (self.some_things_at(destination, Obstacle) or
            not self.is_inbounds(destination))
        if not thing._bump:
            self[thing] = destination
            super().move_to(thing, destination)
        return thing._bump

    def add_thing(self, thing, location=(0, 0)):
        if self.is_inbounds(location):
            super().add_thing(thing, location)

    def is_inbounds(self, location):
        """Checks to make sure that the location is inbounds (within walls if
        we have walls)"""
        x, y = location
        return not (x < self.x_start or x >= self.x_end or y < self.y_start 
                                                        or y >= self.y_end)

    def random_location_inbounds(self, exclude=None):
        """Returns a random location that is inbounds (within walls if we have
        walls)"""
        location = (random.randint(self.x_start, self.x_end),
                    random.randint(self.y_start, self.y_end))
        if exclude is not None:
            while(location == exclude):
                location = (random.randint(self.x_start, self.x_end),
                            random.randint(self.y_start, self.y_end))
        return location

    def add_walls(self):
        """Put walls around the entire perimeter of the grid."""
        for x in range(self.width):
            self.add_thing(Wall(), (x, 0))
            self.add_thing(Wall(), (x, self.height - 1))
        for y in range(self.height):
            self.add_thing(Wall(), (0, y))
            self.add_thing(Wall(), (self.width - 1, y))

        # Updates iteration start and end (with walls).
        self.x_start, self.y_start = (1, 1)
        self.x_end, self.y_end = (self.width - 1, self.height - 1)

    def turn_heading(self, heading, inc):
        """Return the heading to the left (inc=+1) or right (inc=-1) of
        heading."""
        return turn_heading(heading, inc)


class Obstacle(Thing):
    """Something that can cause a bump, preventing an agent from
    moving into the same square it's in."""
    image_filename = 'question.gif'


class Wall(Obstacle):
    image_filename = 'wall.gif'

