# multiAgents.py
# --------------
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
# 
# Project 2
# March 21, 2022
# Authors: Khris Thammavong and Ervin Chhour

from turtle import distance
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Incentivise winning and avoid losing
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')

        # Score of next state
        score = successorGameState.getScore()

        # Separating scared ghosts and active ghosts since pacman should not avoid scared ghosts
        activeGhosts = []
        for index in range(len(newGhostStates)):
            if newScaredTimes[index] == 0:
                activeGhosts.append(newGhostStates[index])
        
        # Distances to closest ghost and food
        distToGhost = min((manhattanDistance(newPos, ghost.getPosition()) for ghost in activeGhosts), default=0)
        distToFood = min((manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()), default=0)

        # We want a bigger distance to a ghost and a smaller distance to food
        scaledGhostDist = -2 * (0 if distToGhost == 0 else (1/distToGhost)) 
        scaledFoodDist = 0 if distToFood == 0 else 1/distToFood 

        # Number of food
        numFood = successorGameState.getNumFood()
        scaledNumFood = -10 * (0 if numFood == 0 else (1/numFood)) 

        # Incentivise capsules
        newCapsules = successorGameState.getCapsules()
        numCapsules = len(newCapsules)
        scaledNumCapsules = -5 * (0 if numCapsules == 0 else (1/numCapsules))

        # Sum up all the different factors for total score
        return score + scaledGhostDist + scaledFoodDist + scaledNumFood + scaledNumCapsules

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        action = self.minimax(gameState, 0, 0)[1]
        return action

    """
    Dispatch function for minimax
    """
    def minimax(self, state, currDepth, index):
        # Terminal state returns the utility
        if state.isWin() or state.isLose():
            return (state.getScore(), None)

        # Choose between max (pacman) and min (each of the ghosts)
        if index == 0:
            return self.max_value(state, currDepth, index)
        else:
            if index <= state.getNumAgents() - 1:
                return self.min_value(state, currDepth, index)
            else:
                if(currDepth + 1 < self.depth): 
                    return self.max_value(state, currDepth+1, 0)
                else:
                    return (self.evaluationFunction(state), None)

    """
    Max-node evaluation
    """        
    def max_value(self, state, currDepth, index):
        value = float('-inf')
        action = None
        for legalAction in state.getLegalActions(index):
            successorState = state.generateSuccessor(index, legalAction) 
            newValue = self.minimax(successorState, currDepth, index+1)[0]
            if newValue > value: 
                value = newValue
                action = legalAction 
            
        return (value, action)

    """
    Min-node evaluation
    """        
    def min_value(self, state, currDepth, index):
        value = float('inf')
        action = None
        for legalAction in state.getLegalActions(index):
            successorState = state.generateSuccessor(index, legalAction)
            newValue = self.minimax(successorState, currDepth, index+1)[0]
            if newValue < value: 
                value = newValue
                action = legalAction 
        return (value, action)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        action = self.alphaBeta(gameState, 0, 0, float('-inf'), float('inf'))[1]
        return action
    
    """
    Dispatch function for alpha-beta pruning
    """
    def alphaBeta(self, state, currDepth, index, alpha, beta):
        # Terminal state returns the utility
        if state.isWin() or state.isLose():
            return (state.getScore(), None)

        # Choose between max (pacman) and min (each of the ghosts)
        if index == 0:
            return self.max_value(state, currDepth, index, alpha, beta)
        else:
            if index <= state.getNumAgents() - 1:
                return self.min_value(state, currDepth, index, alpha, beta)
            else:
                if(currDepth + 1 < self.depth): 
                    return self.max_value(state, currDepth+1, 0, alpha, beta)
                else:
                    return (self.evaluationFunction(state), None)

    """
    Max-node evaluation
    """        
    def max_value(self, state, currDepth, index, alpha, beta):
        value = float('-inf')
        action = None
        for legalAction in state.getLegalActions(index):
            successorState = state.generateSuccessor(index, legalAction) 
            newValue = self.alphaBeta(successorState, currDepth, index+1, alpha, beta)[0]
            if newValue > value: 
                value = newValue
                action = legalAction 
            if value > beta:
                return (value, action)
            alpha = max(alpha, value)
            
        return (value, action)

    """
    Min-node evaluation
    """        
    def min_value(self, state, currDepth, index, alpha, beta):
        value = float('inf')
        action = None
        for legalAction in state.getLegalActions(index):
            successorState = state.generateSuccessor(index, legalAction)
            newValue = self.alphaBeta(successorState, currDepth, index+1, alpha, beta)[0]
            if newValue < value: 
                value = newValue
                action = legalAction 
            if value < alpha:
                return (value, action)
            beta = min(beta, value)
        return (value, action)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        action = self.expectimax(gameState, 0, 0)[1]
        return action
    
    def expectimax(self, state, currDepth, index):
        # Terminal state returns the utility
        if state.isWin() or state.isLose():
            return (state.getScore(), None)

        # Choose between max (pacman) and min (each of the ghosts)
        if index == 0:
            return self.max_value(state, currDepth, index)
        else:
            if index <= state.getNumAgents() - 1:
                return self.expected_value(state, currDepth, index)
            else:
                if(currDepth + 1 < self.depth): 
                    return self.max_value(state, currDepth+1, 0)
                else:
                    return (self.evaluationFunction(state), None)

    """
    Max-node evaluation
    """        
    def max_value(self, state, currDepth, index):
        value = float('-inf')
        action = None
        for legalAction in state.getLegalActions(index):
            successorState = state.generateSuccessor(index, legalAction) 
            newValue = self.expectimax(successorState, currDepth, index+1)[0]
            if newValue > value: 
                value = newValue
                action = legalAction 
            
        return (value, action)

    """
    Expecti-node evaluation
    """        
    def expected_value(self, state, currDepth, index):
        value = 0
        action = None
        probability = 1 / len(state.getLegalActions(index))
        for legalAction in state.getLegalActions(index):
            successorState = state.generateSuccessor(index, legalAction)
            value += probability * self.expectimax(successorState, currDepth, index+1)[0]
            
        return (value, action)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Did the basically the same thing as our other evaluation function w/minor changes to pass the autograder
    We just used the current state rather than the new state given an action
    More details on why we did what we did are in the inline comments
    """
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Incentivise winning and avoid losing
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    # Score of next state
    score = currentGameState.getScore()

    # Separating scared ghosts and active ghosts since pacman should not avoid scared ghosts
    activeGhosts = []
    for index in range(len(ghostStates)):
        if newScaredTimes[index] == 0:
            activeGhosts.append(ghostStates[index])
    
    # Manhattan distances to closest ghost and food
    distToGhost = min((manhattanDistance(position, ghost.getPosition()) for ghost in activeGhosts), default=0)
    distToFood = min((manhattanDistance(position, foodPos) for foodPos in food.asList()), default=0)

    # We want a bigger distance to a ghost and a smaller distance to food
    # so we multiply the ghost distance by a negative factor to avoid being close to ghosts
    scaledGhostDist = -2 * (0 if distToGhost == 0 else (1/distToGhost)) 
    scaledFoodDist = (0 if distToFood == 0 else 1/distToFood) 

    # Number of food, less food = better so we also scale this negatively if pacman doesn't grab enough food
    numFood = currentGameState.getNumFood()
    scaledNumFood = -10 * (0 if numFood == 0 else (1/numFood)) 

    # Sum up all the different factors for the evaluation of the state
    return score + scaledGhostDist + scaledFoodDist + scaledNumFood

# Abbreviation
better = betterEvaluationFunction
