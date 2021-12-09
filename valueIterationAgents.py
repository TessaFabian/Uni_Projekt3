# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # only iterate through the whole mdp tree until reaching self.iterations
        # need a counter starting by zero
        my_iterator = 0

        while my_iterator < self.iterations:
            copy_values = self.values.copy() # we update more than one state in each iteration
            my_iterator += 1
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    poss_actions = self.mdp.getPossibleActions(state)
                    q_values = [self.computeQValueFromValues(state, action) for action in poss_actions]
                    copy_values[state] = max(q_values)
            self.values = copy_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        """
         Q*(s) = all successor states after taken action a from state s have
         to be considered
        """
        successors = self.mdp.getTransitionStatesAndProbs(state, action)
        q_star = 0.0
        # for loop = summation
        for state_next, probability in successors:
            reward = self.mdp.getReward(state, action, state_next)
            q_star += probability * (reward + self.discount*self.getValue(state_next))
        return q_star

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** calculate policy pi(state) ***"
        # for each state and action, compute q value
        #all_q_values = [] --> not possible, argmax from util.py forces to use
        poss_actions = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state):
            # a terminal state has no outgoing actions
            return None
        all_q_values = util.Counter()
        for action in poss_actions:
            all_q_values[action] = self.computeQValueFromValues(state, action)
        return all_q_values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        my_states = self.mdp.getStates()
        my_counter = 0
        for iterator in range(self.iterations):
            state = my_states[iterator % len(my_states)]
            if not self.mdp.isTerminal(state):
                action = self.computeActionFromValues(state)
                if action:
                    self.values[state] = self.computeQValueFromValues(state, action)



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # we need predecessors, an empty priority queue and all states of the given mdp
        predecessors = util.Counter()
        priority = util.PriorityQueue()
        states = self.mdp.getStates()

        def calcPredecessors(states, predecessors):
            # calc predecessors for all states
            # the first state in a mdp has no predecessor
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    newPossibleStates = self.mdp.getTransitionStatesAndProbs(state, action)
                    # newPossibleState = (newState, prob)
                    for entry in newPossibleStates:
                        newState = entry[0]
                        if newState not in predecessors:
                            # each state is a subtree. So we have to begin a new set of predecessors
                            predecessors[newState] = set()
                        predecessors[newState].add(state)
            return predecessors

        def calcNewQValue(state):
            action = self.computeActionFromValues(state)
            return self.computeQValueFromValues(state, action)

        # calc predecessors
        predecessors = calcPredecessors(states, predecessors)

        for state in states:
            if self.mdp.isTerminal(state) == False:
                q_value = calcNewQValue(state)
                diff = abs(q_value - self.values[state])
                priority.push(state, -diff)

        for index in range(self.iterations):
            if priority.isEmpty():
                break
            state = priority.pop()
            newQValue = calcNewQValue(state)
            self.values[state] = newQValue
            preNodesOfState = predecessors[state]
            for predecessor in preNodesOfState:
                newQValueOfPred = calcNewQValue(predecessor)
                diff = abs(newQValueOfPred - self.values[predecessor])
                if diff > self.theta:
                    priority.update(predecessor, -diff)





