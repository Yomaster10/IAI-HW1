from contextlib import closing
from io import StringIO
from os import path
import random
from typing import List, Optional
import time

import numpy as np

import gym
from gym import Env, spaces, utils
from gym.error import DependencyNotInstalled
from typing import List, Tuple



class FrozenLakeEnv(Env):
    """
    Frozen lake involves crossing a frozen lake from Start(S) to Goal(G) without falling into any Holes(H)
    by walking over the Frozen lake.
    ### Action Space
    The agent takes a 1-element vector for actions.
    The action space is `(dir)`, where `dir` decides direction to move in which can be:
    - 0: DOWN
    - 1: RIGHT
    - 2: UP
    - 3: LEFT
    ### Observation Space
    The observation is a value representing the agent's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.
    For example, the 4x4 map has 16 possible observations.
    ### Arguments
    `desc`: Used to specify custom map for frozen lake. For example,
        desc=["SFFF", "FHFH", "FFFH", "HFFG"].
    """

    def __init__(
        self,
        desc
    ):
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.p1 = None
        self.p2 = None

        nA = 4
        nL_cost = {b"F": 10.0, b"H" :0.0, b"T": 3.0, b"A": 2.0, b"L": 1.0, b"S": 1.0, b"G": 1.0, b"P": 100}
        nS = nrow * ncol 

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        for row in range(nrow):
          for col in range(ncol):
            state = self.to_state(row, col)
            if desc[row, col] == b"P":
              if self.p1 == None:
                self.p1 = state
              else:
                self.p2 = state

        for row in range(nrow):
            for col in range(ncol):
                for action in range(nA):
                    new_row, new_col= self.inc(row, col, action)
                    state = self.to_state(row, col)
                    curlleter = desc[row, col]
                    newletter = desc[new_row, new_col]
                    newstate = self.to_state(new_row, new_col)
                    if newletter == b"P":
                        newstate = self.p1 if newstate == self.p2 else self.p2
                    if curlleter == b"H":
                        self.P[state][action] = (None, None, None)
                    else:
                        terminated = bytes(newletter) in b"GH"
                        cost = nL_cost[newletter]
                        self.P[state][action] = (newstate, cost, terminated)          
                  

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = "ansi"
    
    def step(self, a: int) -> Tuple[int, int, bool]:
        """
        Moving the agent one step.

        Args:
            a - action(DOWN, RIGHT, UP, LEFT)
        Returns:
            the new state, the cost of the step and whether the search is over 
            (it can happen when the agent reaches a final state or falls into a hole).
        """
        newstate, cost, terminated = self.P[self.s][a]
        self.s = newstate
        self.lastaction = a
        return (int(newstate), cost, terminated)
    

    def inc(self, row: int, col: int, a: int) -> Tuple[int, int]:
        """
        Given a position and an action, returns the new position.

        Args:
            row - row
            col - col
            a - action
        Returns:
            The new position.
        """
  
        if a == 0:
          row = min(row + 1, self.nrow - 1)
        elif a == 1:
          col = min(col + 1, self.ncol - 1)
        elif a == 2:
          row = max(row - 1, 0)
        elif a == 3:
          col = max(col - 1, 0)
        return (row, col)


    def to_state(self, row: int, col: int) -> int:
        """
        Converts between location on the board and state.
        Args:
            row
            col
        Returns:
            state
        """
        return row * self.ncol + col

    def to_row_col(self, state: int) -> Tuple[int, int]:
        """
        Converts between state and location on the board.
        Args:
            state
        Returns:
            row, col
        """
        return (state // self.ncol, state % self.ncol)

    def succ(self, state: int):
        """
        Returns the successors of the state.
        Args:
            state
        Returns:
            Returns a dictionary that contains information on all the successors of the state.
            The keys are the actions. 
            The values are tuples of the form (new state, cost, terminated). 
            Note that terminated is true when the agent reaches a final state or a hole.
        """
        return self.P[state]
    
    def set_state(self, state: int) -> None:
        """
        Sets the current state of the agent.
        """
        self.s = state

    def get_state(self):
        """
        Returns the current state of the agent.
        """
        return self.s

    def is_final_state(self, state: int) -> bool:
        """
        Retruens if the state is a final state.
        The function can help you understand whether you have fallen
        into a hole or whether you have reached a final state
        """
        return (self.ncol * self.nrow - 1) == state

    def get_initial_state(self) -> int:
        """
        Retruens the initial state.
        """
        return 0


    def reset(self) -> int:
        """
        Initializes the search problem. 
        """
        super().reset()
        self.s = self.get_initial_state()
        self.lastaction = None

        return int(self.s)



    def render(self):
        """
        Returns a view of the board. 
        """
        return self._render_text()

    
    def _render_text(self):
        desc = self.desc.copy().tolist()

        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        for r in range(len(desc)):
            for c in range(len(desc[r])):
                if desc[r][c] == "H":
                    desc[r][c] = utils.colorize(desc[r][c], "white", highlight=True)
                elif desc[r][c] == "P":
                    desc[r][c] = utils.colorize(desc[r][c], "green", highlight=True)
                elif desc[r][c] == "G":
                    desc[r][c] = utils.colorize(desc[r][c], "yellow", highlight=True)
                elif desc[r][c] in "FTAL":
                    desc[r][c] = utils.colorize(desc[r][c], "blue", highlight=True)
                else:
                    desc[row][col] = utils.colorize(desc[row][col], "magenta", highlight=True)
        desc[0][0] = utils.colorize(desc[0][0], "yellow", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Down', 'Right', 'Up', 'Left'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()
