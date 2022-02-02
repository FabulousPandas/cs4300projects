# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# Project 1
# February 7, 2022
# Authors: Khris Thammavong and Ervin Chhour

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from inspect import stack
from tabnanny import check
from tracemalloc import start
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]
"""
Given the problem, the appropriate data structure, and a function corresponding to the algorithm
the function performs its corresponding search and returns a list of actions to the goal if it reaches it
"""
def generalSearchAlgorithm(problem, dataStructure):
    seenSet = set()
    startState = (problem.getStartState(), None, 0, [])
    goal = None

    dataStructure.push(startState)
    while not dataStructure.isEmpty():
        node = dataStructure.pop()
        if problem.isGoalState(node[0]):
            goal = node
            break
        if not node[0] in seenSet:
            seenSet.add(node[0])
            for successor in problem.getSuccessors(node[0]):
                actionList = list(node[3])
                actionList.append(successor[1])
                newNode = (successor[0], successor[1], successor[2], actionList)
                dataStructure.push(newNode)
    if not goal == None:
        return goal[3]
    return []

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """
    stack = util.Stack()
    return generalSearchAlgorithm(problem, stack)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    return generalSearchAlgorithm(problem, queue)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    pq = util.PriorityQueueWithFunction(lambda item: problem.getCostOfActions(item[3]))
    return generalSearchAlgorithm(problem, pq)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    pq = util.PriorityQueueWithFunction(lambda item : problem.getCostOfActions(item[3]) + heuristic(item[0], problem))
    return generalSearchAlgorithm(problem, pq)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
