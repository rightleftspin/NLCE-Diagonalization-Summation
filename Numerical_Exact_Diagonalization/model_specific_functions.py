#!/usr/bin/env python3
from itertools import product
# It might be useful to put a dictionary here that has a key for each model and inside the key is all the associated functions that one needs to solve that model.

def empty(site, state):
    return([])

def ising_kinetic(bond, state):
    # Simple kinetic term for the ising model, just uses the bond type to
    # determine the amount of energy
    final_state = []
    j = -1
    if (state[bond[0]] == state[bond[1]]):
        final_state.append((j * (1 / bond[2]), hash(state)))
    else:
        final_state.append((-j * (1 / bond[2]), hash(state)))
    
    return(final_state)

def heisenberg_kinetic(bond, state):
    final_state = []
    j = -1
    if (state[bond[0]] == state[bond[1]]):
        final_state.append((j * (1 / bond[2]), hash(state)))
    else: 
        final_state.append((-j * (1 / bond[2]), hash(state)))
        new_state = list(state)
        new_state[bond[0]] = not state[bond[0]]
        new_state[bond[1]] = not state[bond[1]]
        final_state.append((-j * 0.5 * (1 / bond[2]), hash(tuple(new_state))))

    return(final_state)

def generate_states_hubbard(num_sites, num_particles_spin_sep):
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

def generate_states_ising_heisenberg(num_sites, num_particles_spin_sep):
    # Doesn't need the number of particles seperated by spin like the
    # Hubbard model, but helps with consistency
    # Generate the ising model states
    all_possible_states = list(product([True, False], repeat=num_sites))

    return(all_possible_states)

model_info = {
    "ising": {
        "generator": generate_states_ising_heisenberg,
        "bond_solver": ising_kinetic,
        "site_solver": empty
    },
    "heisenberg": {
        "generator": generate_states_ising_heisenberg,
        "bond_solver": heisenberg_kinetic,
        "site_solver": empty
    }
}
