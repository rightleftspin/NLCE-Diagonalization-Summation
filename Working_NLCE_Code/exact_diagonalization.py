import dask
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.linalg as sl
from tqdm import tqdm
from dask.diagnostics import ProgressBar

def get_bit(value, n):
    return ((value >> n & 1) != 0)

def set_bit(value, n):
    return value | (1 << n)

def clear_bit(value, n):
    return value & ~(1 << n)

def flip_bit(value, n):
    return value ^ (1 << n)

def swap_bits(value, n, m):

    # Move p1'th to rightmost side
    bit1 = (value >> n) & 1

    # Move p2'th to rightmost side
    bit2 = (value >> m) & 1

    # XOR the two bits
    x = (bit1 ^ bit2)

    # Put the xor bit back to their original positions
    x = (x << n) | (x << m)

    # XOR 'x' with the original number so that the
    # two sets are swapped
    result = value ^ x
    return result


def eig_to_property_general(property_array, energy_array, temp_grid, number_sites):
    """
    General purpose summation for a general property
    """
    
    exp_energy_temp_matrix = np.exp(-energy_array[:, np.newaxis] / temp_grid)
    partition_function = exp_energy_temp_matrix.sum(axis=0)
    prop_sum = np.matmul(property_array, exp_energy_temp_matrix)

    final_prop = prop_sum / partition_function

    return(final_prop)

def get_free_energy(energy_array, temp_grid):
    """
    Returns partition function of given energy eigenvalues
    """
    exp_energy_temp_matrix = np.exp(-energy_array[:, np.newaxis] / temp_grid)
    partition_function = exp_energy_temp_matrix.sum(axis=0)

    return(np.log(partition_function))

@dask.delayed
def find_ising_energy_eigenvalues(bond_information, number_sites, tunneling_strength):
    """
    This function takes the bond information for a specific graph
    and returns its ising energy eigenvalues
    """
    number_states = 2 ** number_sites
    eigenvalues = []
    mag = []

    h = 0
    for state in range(number_states):
        e_state = 0
        for bond in bond_information:
            if get_bit(state, bond[0]) == get_bit(state, bond[1]):
                e_state += tunneling_strength[bond[2] - 1]
            else:
                e_state -= tunneling_strength[bond[2] - 1]

        net_spin = (2 * (bin(state).replace("0b", "").count('1'))) - number_sites
        mag.append(net_spin)
        eigenvalues.append(e_state - (h * net_spin))

    return(number_sites, np.array(eigenvalues), np.array(mag))

def ising_all(input_dict, graph_bond_info_ordered):
    """
    Takes in the graph bond information and returns
    the property dictionary unordered
    """
    benchmarking = input_dict["benchmarking"]
    temp_range = input_dict["temp_range"]
    grid_granularity = input_dict["grid_granularity"]
    tunneling_strength = input_dict["tunneling_strength"]

    temp_grid = np.logspace(temp_range[0], temp_range[1], num = grid_granularity)
    eig_dict = {}
    property_dict = {"Energy": {}, "Specific Heat": {}, "Magnetization": {}, "Susceptibility": {}, "Free Energy": {}}

    for order, graph_bond_info in graph_bond_info_ordered.items():
        eig_dict.update({k: find_ising_energy_eigenvalues(v, order, tunneling_strength) for k, v in graph_bond_info.items()})

    if benchmarking:
        pbar = ProgressBar()
        pbar.register()

    eig_dict_solved = dask.compute(eig_dict, scheduler = "processes", num_workers=input_dict["number_cores"])[0]

    if benchmarking:
        start = time.time()
        eig_dict_items = tqdm(eig_dict_solved.items())
        print("Starting Property Solving")
    else:
        eig_dict_items = eig_dict_solved.items()

    for graph_id, eig_vals in eig_dict_items:
        number_sites, energy, magnetization = eig_vals
        property_dict["Energy"][graph_id] = eig_to_property_general(energy, energy, temp_grid, number_sites)
        property_dict["Specific Heat"][graph_id] = (eig_to_property_general(energy ** 2, energy, temp_grid, number_sites) - (property_dict["Energy"][graph_id] ** 2)) / (temp_grid ** 2) 

        property_dict["Magnetization"][graph_id] = eig_to_property_general(magnetization, energy, temp_grid, number_sites)
        property_dict["Susceptibility"][graph_id] = (eig_to_property_general(magnetization ** 2, energy, temp_grid, number_sites) - (property_dict["Magnetization"][graph_id] ** 2)) / (temp_grid) 

        property_dict["Free Energy"][graph_id] = get_free_energy(energy, temp_grid)
        
    if benchmarking:
        print(f"Finished property solving in {time.time() - start:.4f}s")

    return(property_dict, temp_grid)

@dask.delayed
def find_heisenberg_eigenvalues(bond_information, number_sites, tunneling_strength):
    """
    This function takes the bond information for a specific graph
    and returns its heisenberg energy eigenvalues
    """
    number_states = 2 ** number_sites
    hamiltonian_matrix = np.zeros((numbers_states, number_states))
    mag = []

    h = 0
    for state in range(number_states):
        for bond in bond_information:
            site1, site2, bond_strength = get_bit(state, bond[0]), get_bit(state, bond[1]), tunneling_strength[bond[2]]
            if site1 == site2:
                hamiltonian_matrix[state, state] += 0.25 * bond_strength
            else:
                hamiltonian_matrix[state, state] -= 0.25 * bond_strength
                new_state = flip_bit(flip_bit(state, site1), site2)
                hamiltonian_matrix[state, new_state] = 0.5 * bond_strength
        net_spin = (2 * (bin(state).replace("0b", "").count('1'))) - number_sites
        mag.append(net_spin)

    eigenvalues = sl.eigh(hamiltonian_matrix, eigvals_only = True)

    return(number_sites, eigenvalues, np.array(mag))

def heisenberg_all(input_dict, graph_bond_info_ordered):
    """
    Takes in the graph bond information and returns
    the property dictionary unordered
    """
    benchmarking = input_dict["benchmarking"]
    temp_range = input_dict["temp_range"]
    grid_granularity = input_dict["grid_granularity"]
    tunneling_strength = input_dict["tunneling_strength"]

    temp_grid = np.logspace(temp_range[0], temp_range[1], num = grid_granularity)
    eig_dict = {}
    property_dict = {"Energy": {}, "Specific Heat": {}, "Magnetization": {}, "Susceptibility": {}, "Free Energy": {}}

    for order, graph_bond_info in graph_bond_info_ordered.items():
        eig_dict.update({k: find_ising_energy_eigenvalues(v, order, tunneling_strength) for k, v in graph_bond_info.items()})

    if benchmarking:
        pbar = ProgressBar()
        pbar.register()

    eig_dict_solved = dask.compute(eig_dict, scheduler = "processes", num_workers=input_dict["number_cores"])[0]

    if benchmarking:
        start = time.time()
        eig_dict_items = tqdm(eig_dict_solved.items())
        print("Starting Property Solving")
    else:
        eig_dict_items = eig_dict_solved.items()

    for graph_id, eig_vals in eig_dict_items:
        number_sites, energy, magnetization = eig_vals
        property_dict["Energy"][graph_id] = eig_to_property_general(energy, energy, temp_grid, number_sites)
        property_dict["Specific Heat"][graph_id] = (eig_to_property_general(energy ** 2, energy, temp_grid, number_sites) - (property_dict["Energy"][graph_id] ** 2)) / (temp_grid ** 2)

        property_dict["Magnetization"][graph_id] = eig_to_property_general(magnetization, energy, temp_grid, number_sites)
        property_dict["Susceptibility"][graph_id] = (eig_to_property_general(magnetization ** 2, energy, temp_grid, number_sites) - (property_dict["Magnetization"][graph_id] ** 2)) / (temp_grid)

        property_dict["Free Energy"][graph_id] = get_free_energy(energy, temp_grid)

    if benchmarking:
        print(f"Finished property solving in {time.time() - start:.4f}s")

    return(property_dict, temp_grid)

property_functions = {"ising": ising_all,
                      "heisenberg": heisenberg_all}
