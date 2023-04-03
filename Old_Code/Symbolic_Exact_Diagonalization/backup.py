# Run in Python with NetworkX
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from copy import deepcopy
from itertools import combinations

def raise_lower_state(state, node, spin, raise_or_lower):
    raise_lower_possibility = raise_or_lower ^ state.nodes[node][spin]
    if raise_lower_possibility:
        new_state = deepcopy(state)
        new_state.nodes[node][spin] = raise_or_lower
        return(new_state)
    else:
        return(0)

def raise_lower_pair(state, node1, node2, spin):
    lower_2 = raise_lower_state(state, node2, spin, False)
    if not (lower_2 == 0):
        raise_1 = raise_lower_state(lower_2, node1, spin, True)
        return(raise_1)
    else:
        return(0)

def count_particles(state, node, possible_spin_states):
    spins = map(state.nodes[node].get, possible_spin_states)
    number_of_particles = sum(map(int,spins))
    # This part doesn't generalize to more than 2 spin states, but it can be made to
    if number_of_particles == 2:
        return(1)
    else:
        return(0)

def kinetic_hamiltonian(state, node1, node2, possible_spin_states, strength_keyword):
    strength = state[node1][node2][strength_keyword]
    final_states = []
    for spin_state in possible_spin_states:
        final_states.append((strength, raise_lower_pair(state, node1, node2, spin_state)))
        final_states.append((strength, raise_lower_pair(state, node2, node1, spin_state)))
    
    return(final_states)

def potential_hamiltonian(state, node, possible_spin_states, strength):
    num_particles = count_particles(state, node, possible_spin_states)
    return([((strength * num_particles), state)])

def full_hamiltonian(state, possible_spin_states, transmission_strength_keyword, potential_strength):
    # Kinetic energy terms
    final_state = []
    for connection in state.edges:
        final_state += kinetic_hamiltonian(state, connection[0], connection[1], possible_spin_states, transmission_strength_keyword)

    for node in state.nodes:
        final_state += potential_hamiltonian(state, node, possible_spin_states, potential_strength)

    return(final_state)
    
def state_equality(state1, state2):
    return(nx.utils.graphs_equal(state1, state2))

def dot_with_other_state(right_states, left_state):
    result = 0
    for state_pair in right_states:
        if not((state_pair[0] == 0) or (state_pair[1] == 0)):
            if state_equality(state_pair[1], left_state):
                result += state_pair[0]

    return(result)

# Generate Transmission for 1 node linear case with one up electron in node one and one down electron in node two
possible_spin_states = ["1/2", "-1/2"]
t = Symbol('t')
u = Symbol('u')
sampleLattice = nx.Graph()
nodeList = [ ( 1, {"1/2": True, "-1/2": False} ), ( 2, {"1/2": False, "-1/2": True} ) ]
sampleLattice.add_nodes_from(nodeList)
edgeList = [ (1, 2, {"strength" : -t}) ]
sampleLattice.add_edges_from(edgeList)

finalStates = full_hamiltonian(sampleLattice, possible_spin_states, "strength", u)

newLattice = nx.Graph()
nodeList = [ ( 1, {"1/2": False, "-1/2": False} ), ( 2, {"1/2": True, "-1/2": True} ) ]
newLattice.add_nodes_from(nodeList)
edgeList = [ (1, 2, {"strength" : -t}) ]
newLattice.add_edges_from(edgeList)

print(dot_with_other_state(finalStates, newLattice))

## There is an is_isomorphic function in this toolkit that will tell you if two graphs are isomorphic, which will be particularly useful to you
## at some point when dealing with symmetries. make sure to check it out later
