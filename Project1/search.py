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
def generalSearchAlgorithm(problem, dataStructure, updateDataStructure):
    actionList = list()
    seenSet = set()
    parentMap = dict()
    startState = (problem.getStartState(), None, 0)
    # print(startState)
    goal = None

    dataStructure.push(startState)
    while not dataStructure.isEmpty():
        node = dataStructure.pop()
        # print("node: " + str(node))
        if problem.isGoalState(node[0]):
            goal = node
            break
        if not node[0] in seenSet:
            # print("dataStructure: " + str(dataStructure))
            # print("node: " + str(node))
            seenSet.add(node[0])
            for successor in problem.getSuccessors(node[0]):
                updateDataStructure(node, dataStructure, successor, seenSet, parentMap)

    node = goal
    # print("goal: " + str(node))
    # print("")
    # print(parentMap)
    # print("")
    # print("first argument: " + str(not node == None) + " second argument: " + str(node[0] in parentMap))
    while not node == None and not node[0] == startState[0]:
        # print("adding to actionList, node = " + str(node) )
        actionList.append(parentMap[node[0]][1])
        node = parentMap[node[0]][0]
    actionList.reverse() 
    # print(actionList)
    return actionList

"""
Update function for DFS
"""
def dfsUpdate(node, dataStructure, successor, seenSet, parentMap):
    dataStructure.push(successor)
    if successor[0] not in seenSet:
        parentMap[successor[0]] = (node, successor[1])

"""
Update function for BFS
"""
def bfsUpdate(node, dataStructure, successor, seenSet, parentMap):
    dataStructure.push(successor)
    if successor[0] not in seenSet and successor[0] not in parentMap:
        parentMap[successor[0]] = (node, successor[1])
"""
Update function for UCS
"""
def ucsUpdate(node, dataStructure, successor, seenSet, parentMap):
    dataStructure.push(successor)
    # print("in update: successor = " + str(successor))
    if(successor[0] not in parentMap):
        parentMap[successor[0]] = (node, successor[1], successor[2])
    elif (successor[2] < parentMap[successor[0]][2]):
        parentMap[successor[0]] = (node, successor[1], successor[2])

def getCostFromItem(item):
    return item[2]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """

    stack = util.Stack()
    return generalSearchAlgorithm(problem, stack, dfsUpdate)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    return generalSearchAlgorithm(problem, queue, bfsUpdate)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # return []
    pq = util.PriorityQueueWithFunction(getCostFromItem)
    return generalSearchAlgorithm(problem, pq, ucsUpdate)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
