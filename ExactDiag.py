#!/usr/bin/env python3
from itertools import product
import numpy as np
import scipy
import json
import multiprocessing as mp
from functools import partial
import pathlib

# This file contains all the functions nessecary to start with a cluster from the NLCE output (number of sites and bond information)
# and will return to you the hamiltonian matrix corresponding to the number of particles you want, it should work with any spin types,
# electrons are the only ones that we really want to analyze right now. To add a new model simply define two functions,
# 1) how does the model deal with bond information
# 2) how does the model deal with site information
# using these, this algorithm can generate the hamiltonian matrix

def ising_kinetic(bond, state):
    # Simple kinetic term for the ising model, just uses the bond type to
    # determine the amount of energy
    final_state = []
    if (state[bond[0]] == state[bond[1]]):
        final_state.append((-bond[2], hash(state)))
    else:
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
        final_state += ising_kinetic(bond, state)

   # for site in range(num_sites):
   #     site_info = [spin_state[site] for spin_state in state]
   #     final_state += []

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

def generate_states_ising(num_sites):
    all_possible_states = list(product([True, False], repeat=num_sites))

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

def solve_energies(num_sites, num_particles_spin_sep, bond_info):

    all_possible_states = generate_states_ising(num_sites)
    hamil_matrix = generate_hamil_matrix(num_sites, num_particles_spin_sep, bond_info, all_possible_states)

    eigenvals = hamil_matrix.diagonal()
    #eigenvals = scipy.linalg.eigvals(hamil_matrix).real

    return(eigenvals)

def energy_solver(graph_id, order, graph_bond_dict, temperature_array):
    graph_property_info = {}
    bond_info = graph_bond_dict[graph_id]
    energies = solve_energies(order, [], bond_info)

    exp_energy_temp_matrix = np.exp(-energies[:, np.newaxis] / temperature_array)
    partition_function = exp_energy_temp_matrix.sum(axis=0)
    energy = np.matmul(energies, exp_energy_temp_matrix)

    final_energies = energy / partition_function

    graph_property_info[graph_id] = list(final_energies)

    return(graph_property_info)

def specific_heat_solver(graph_id, order, graph_bond_dict, temperature_array):
    graph_property_info = {}
    bond_info = graph_bond_dict[graph_id]
    energies = solve_energies(order, [], bond_info)

    exp_energy_temp_matrix = np.exp(-energies[:, np.newaxis] / temperature_array)
    partition_function = exp_energy_temp_matrix.sum(axis=0)
    energy = np.matmul(energies, exp_energy_temp_matrix)
    energy_sq = np.matmul(energies ** 2, exp_energy_temp_matrix)

    final_energies = (energy / partition_function) ** 2
    final_energies_sq = energy_sq / partition_function

    energy_unc = final_energies_sq - final_energies
    specific_heat = energy_unc / (temperature_array ** 2)
    graph_property_info[graph_id] = list(specific_heat)

    return(graph_property_info)


def solve_property_for_order(property_function, property_name, data_dir, order, nlce_type, temperature_array):

    graph_bond = open(f'{data_dir}/graph_bond_{nlce_type}_{order}.json')
    graph_bond_dict = json.load(graph_bond)

    if not pathlib.Path(f"{data_dir}/graph_{property_name}_info_{nlce_type}_{order}.json").exists():
        property_solve_graph = partial(property_function, order = order, graph_bond_dict = graph_bond_dict, temperature_array = temperature_array)

        # Parallellize here
        cpus = mp.cpu_count()
        pool = mp.Pool(cpus)
        graph_property_list = list(pool.map(property_solve_graph, graph_bond_dict.keys()))
        graph_property_info = {}

        for graph in graph_property_list:
            graph_property_info.update(graph)

        graph_property_info_json = open(f"{data_dir}/graph_{property_name}_info_{nlce_type}_{order}.json", "w")
        json.dump(graph_property_info, graph_property_info_json)
    else:
        graph_property_info_json = open(f"{data_dir}/graph_{property_name}_info_{nlce_type}_{order}.json")
        graph_property_info = json.load(graph_property_info_json)

    return(graph_property_info)

#def solve_specific_heat_for_order_slow(data_dir, order, nlce_type, temp_range, granularity):
#
#    temperature_array = np.logspace(temp_range[0], temp_range[1], num=granularity)
#
#    graph_property_info = {}
#
#    graph_bond = open(f'{data_dir}/graph_bond_{nlce_type}_{order}.json')
#    graph_bond_dict = json.load(graph_bond)
#
#    # Parallellize here
#    for graph_id in graph_bond_dict:
#        bond_info = graph_bond_dict[graph_id]
#        energies = solve_energies(order, [], bond_info)
#
#        exp_energy_temp_matrix = np.exp(-energies[:, np.newaxis] / temperature_array)
#        partition_function = exp_energy_temp_matrix.sum(axis=0)
#        energy = np.matmul(energies, exp_energy_temp_matrix)
#        energy_sq = np.matmul(energies ** 2, exp_energy_temp_matrix)
#
#        final_energies = (energy / partition_function) ** 2
#        final_energies_sq = energy_sq / partition_function
#
#        energy_unc = final_energies_sq - final_energies
#        specific_heat = energy_unc / (temperature_array ** 2)
#        graph_property_info[graph_id] = list(specific_heat)
#
#
#    graph_property_info_json = open(f"{data_dir}/graph_energy_info_{nlce_type}_{order}.json", "w")
#    json.dump(graph_property_info, graph_property_info_json)
#
#    return(graph_property_info)
