#!/usr/bin/env python3
import numpy as np
import scipy

def hamiltonian(bond_solver, site_solver, state, num_sites, num_spin_states, bond_info):
    # Takes in a specific state and returns a list of tuples of the form
    # (coefficient, new_state)
    # state is in the form [(True, False, True, ...), (True, False, False, ...)] corresponding to
    # spin up and spin down respectively
    # bond info is in the form [(1, 0, 1), (2, 3, 1), (4, 1, 2), ...] which tells the algorithm
    # about the connections (first two numbers) and the bond strength (third number)
    final_state = []
    for bond in bond_info:
        # Deals generally with the "kinetic" terms, largely interaction over bonds
        final_state += bond_solver(bond, state)

    for site in range(num_sites):
        # Deals generally with the "potential" terms, largerly interactions within the site
        final_state += site_solver(site, state)

    return(final_state)

def generate_hamil_matrix(bond_solver, site_solver, num_sites, num_particles_spin_sep, bond_info, all_possible_states):

    all_possible_states_hashed = list(map(hash, all_possible_states))

    hamiltonian_filled = lambda state: hamiltonian(bond_solver, site_solver, state, num_sites, len(num_particles_spin_sep), bond_info)

    hamil_matrix = np.zeros((len(all_possible_states), len(all_possible_states)))

    for n, state in enumerate(all_possible_states):
        states_acted_on = hamiltonian_filled(state)
        for m, hashed_state in enumerate(all_possible_states_hashed):
            for state_acted_on in states_acted_on:
                if state_acted_on[1] == hashed_state:
                    hamil_matrix[n][m] += state_acted_on[0]

                
    return(hamil_matrix)

def solve_for_property(state_generator, bond_solver, site_solver, num_sites, num_particles_spin_sep, bond_info):
    all_possible_states = state_generator(num_sites, num_particles_spin_sep)
    hamil_matrix = generate_hamil_matrix(bond_solver, site_solver, num_sites, num_particles_spin_sep, bond_info, all_possible_states)
    eigenvals = scipy.linalg.eigh(hamil_matrix, eigvals_only = True)
    return(eigenvals)
