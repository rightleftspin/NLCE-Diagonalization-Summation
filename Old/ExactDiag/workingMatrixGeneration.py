# Run in Python with NetworkX
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from copy import deepcopy
from itertools import combinations, product

def add_list_particles(lattice, list_particles, spin_state):
    new_lattice = deepcopy(lattice)
    for node in list_particles:
        new_lattice.nodes[node][spin_state] = True

    return(new_lattice)

def generate_all_states(empty_lattice, spin_states, num_each_spin_state):
    all_possible_states = []
    all_combinations = [list(combinations(empty_lattice.nodes, particles)) for particles in num_each_spin_state]
    cart_product_combos = product(*all_combinations)
    for state in list(cart_product_combos):
        empty_state = deepcopy(empty_lattice)
        for index, configuration in enumerate(state):
            empty_state = add_list_particles(empty_state, configuration, spin_states[index])
        all_possible_states.append(empty_state)

    return(all_possible_states)

def add_empty_particles(lattice, spin_states):
    copied_lattice = deepcopy(lattice)
    for node in copied_lattice.nodes:
        for spin_state in spin_states:
            copied_lattice.nodes[node][spin_state] = False 

    return(copied_lattice)

def give_connections_strength(lattice, transmission_keyword, transmission_strength):
    new_lattice = deepcopy(lattice)
    for edge in new_lattice.edges:
        new_lattice.edges[edge][transmission_keyword] = transmission_strength

    return(new_lattice)


def add_transmission_strength(lattice, strength, strength_keyword):
    copied_lattice = deepcopy(lattice)
    for connection in state.edges:
        copied_lattice.edges[connection][strength_keyword] = strength

    return(copied_lattice)

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

def generate_full_hamiltonian_matrix(lattice, possible_spin_states, num_each_spin_state, transmission_keyword, transmission_strength, potential_strength):
    empty_lattice = add_empty_particles(lattice, possible_spin_states)
    empty_lattice = give_connections_strength(empty_lattice, transmission_keyword, transmission_strength)
    all_possible_states = generate_all_states(empty_lattice, possible_spin_states, num_each_spin_state)
    num_states = len(all_possible_states)
    hamil_matrix = zeros(num_states, num_states) 
    for row, right_state in enumerate(all_possible_states):
        acted_on = full_hamiltonian(right_state, possible_spin_states, transmission_keyword, potential_strength) 
        for column, left_state in enumerate(all_possible_states):
            hamil_matrix[row, column] = dot_with_other_state(acted_on, left_state)

    return(hamil_matrix)




possible_spin_states = ["1/2", "-1/2"]
num_each_spin_state = [1, 1]
t = Symbol('t')
u = Symbol('u')
sampleLattice = nx.Graph()
nodeList = [ 1, 2 ]
sampleLattice.add_nodes_from(nodeList)
edgeList = [ (1, 2) ]
sampleLattice.add_edges_from(edgeList)
print(len(generate_all_states(sampleLattice, possible_spin_states, num_each_spin_state)))
#pprint(generate_full_hamiltonian_matrix(sampleLattice, possible_spin_states, num_each_spin_state, 'strength', -t, u))

#squareLattice = nx.grid_2d_graph(3, 3)
#spin_states = ["1/2", "-1/2"]
#new_lattice = add_empty_particles(squareLattice, spin_states)
#for state, graph in enumerate(generate_all_states(new_lattice, possible_spin_states, num_each_spin_state)):
#    print(state)
#    for node in graph.nodes:
#        print(f"node {node} has up: {graph.nodes[node]['-1/2']} down:{graph.nodes[node]['1/2']}")
#
#
#hexLattice = nx.hexagonal_lattice_graph(3,3)
