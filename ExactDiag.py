#!/usr/bin/env python3
from itertools import product
import numpy as np

# This file contains all the functions nessecary to start with a cluster from the NLCE output (number of sites and bond information)
# and will return to you the hamiltonian matrix corresponding to the number of particles you want, it should work with any spin types,
# electrons are the only ones that we really want to analyze right now. To add a new model simply define two functions,
# 1) how does the model deal with bond information
# 2) how does the model deal with site information
# using these, this algorithm can generate the hamiltonian matrix

def ising_kinetic(bond, state, num_spin_states):
    # Simple kinetic term for the ising model, just uses the bond type to
    # determine the amount of energy
    final_state = []
    for spin_state in range(num_spin_states):
        if (state[spin_state][bond[0]] and state[spin_state][bond[1]]):
            final_state.append((bond[2], hash(state)))
    
    return(final_state)


def hamiltonian(state, num_sites, num_spin_states, bond_info):
    # Takes in a specific state and returns a list of tuples of the form
    # (coefficient, new_state)
    # state is in the form [(True, False, True, ...), (True, False, False, ...)] corresponding to
    # spin up and spin down respectively
    # bond info is in the form [(1, 0, 1), (2, 3, 1), (4, 1, 2), ...] which tells the algorithm
    # about the connections (first two numbers) and the bond strength (third number)
    final_state = []
    for bond in bond_info:
        final_state += ising_kinetic(bond, state, num_spin_states)

   # for site in range(num_sites):
   #     site_info = [spin_state[site] for spin_state in state]
   #     final_state += []

    return(final_state)

def generate_states(num_sites, num_particles_spin_sep):
    # Takes a number of sites and number of particles per spin type and returns two lists
    # one of all possible states and one of all these states hashed
    num_spin_states = len(num_particles_spin_sep)
    state_checker_full = lambda state, num_particles: sum(state) == num_particles

    all_possible_states_spin_sep = []
    for num_particles in num_particles_spin_sep:
        single_spin = product([True, False], repeat=num_sites)
        state_checker = lambda state: state_checker_full(state, num_particles)
        all_possible_states_spin_sep.append(list(filter(state_checker, single_spin)))

    all_possible_states = list(product(*all_possible_states_spin_sep))

    return(all_possible_states)

def generate_hamil_matrix(num_sites, num_particles_spin_sep, bond_info, all_possible_states):

    all_possible_states_hashed = list(map(hash, all_possible_states))

    hamiltonian_filled = lambda state: hamiltonian(state, num_sites, len(num_particles_spin_sep), bond_info)

    hamil_matrix = np.zeros((len(all_possible_states), len(all_possible_states)))

    for n, state in enumerate(all_possible_states):
        states_acted_on = hamiltonian_filled(state)
        for m, hashed_state in enumerate(all_possible_states_hashed):
            for state_acted_on in states_acted_on:
                if state_acted_on[1] == hashed_state:
                    hamil_matrix[n][m] += state_acted_on[0]

                
    return(hamil_matrix)

num_sites = 3
num_particles_spin_sep = [2, 1]
bond_info_example = [(0, 1, 1), (1, 2, 1)]

all_possible_states = generate_states(num_sites, num_particles_spin_sep)
hamil_matrix = generate_hamil_matrix(num_sites, num_particles_spin_sep, bond_info_example, all_possible_states)
