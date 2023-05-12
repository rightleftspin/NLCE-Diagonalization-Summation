import dask
import time
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.linalg as sl
from tqdm import tqdm
from dask.diagnostics import ProgressBar

np.set_printoptions(threshold=sys.maxsize)

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
    temp_2d, energy_2d= np.meshgrid(temp_grid, energy_array)
    exp_energy_temp_matrix = np.exp(-energy_2d / temp_2d)
    partition_function = exp_energy_temp_matrix.sum(axis=0)
    prop_sum = np.matmul(property_array, exp_energy_temp_matrix)

    final_prop = prop_sum / partition_function
    return(final_prop)

def eig_to_property_3d(property_array, magnetization, energy_array, mag_grid, temp_grid, number_sites):
    """
    General purpose summation for a general property with magnetic field
    """
    temp_3d, mag_3d, energy_3d = np.meshgrid(temp_grid, mag_grid, energy_array)
    energy_3d = energy_3d - (0.5 * (mag_3d * magnetization))
    exp_energy_temp_matrix = np.exp(-energy_3d / temp_3d)
    partition_function = exp_energy_temp_matrix.sum(axis=2)
    prop_sum = np.tensordot(exp_energy_temp_matrix, property_array, axes = ([2, 0]))

    final_prop = prop_sum / partition_function
    return(final_prop)

def free_energy_3d(magnetization, energy_array, mag_grid, temp_grid):
    """
    General purpose summation for a general property with magnetic field
    """
    temp_3d, mag_3d, energy_3d = np.meshgrid(temp_grid, mag_grid, energy_array)
    energy_3d = energy_3d - (0.5 * (mag_3d * magnetization))
    exp_energy_temp_matrix = np.exp(-energy_3d / temp_3d)
    partition_function = exp_energy_temp_matrix.sum(axis=2)

    return(np.log(partition_function))

def get_free_energy(energy_array, temp_grid):
    """
    Returns partition function of given energy eigenvalues
    """
    exp_energy_temp_matrix = np.exp(-energy_array[:, np.newaxis] / temp_grid)
    partition_function = exp_energy_temp_matrix.sum(axis=0)

    return(np.log(partition_function))

@dask.delayed
def find_ising_properties(input_dict, graph_id, bond_information, number_sites):
    """
    This function takes the bond information for a specific graph
    and returns its ising energy eigenvalues
    Uses spin convention where spin operator returns 1/4
    """
    benchmarking = input_dict["benchmarking"]
    temp_range = input_dict["temp_range"]
    grid_granularity = input_dict["grid_granularity"]
    tunneling_strength = input_dict["tunneling_strength"]
    mag_range = input_dict["mag_range"]
    mag_granularity = input_dict["mag_granularity"]

    number_states = 2 ** number_sites
    eigenvalues = []
    mag = []

    for state in range(number_states):
        e_state = 0
        for bond in bond_information:
            if get_bit(state, bond[0]) == get_bit(state, bond[1]):
                e_state += 0.25 * tunneling_strength[bond[2] - 1]
            else:
                e_state -= 0.25 * tunneling_strength[bond[2] - 1]

        net_spin = (2 * (bin(state).replace("0b", "").count('1'))) - number_sites
        mag.append(net_spin)
        eigenvalues.append(e_state)

    eigenvalues, mag = np.array(eigenvalues), np.array(mag)

    temp_grid = np.logspace(temp_range[0], temp_range[1], num = grid_granularity)
    mag_grid = np.linspace(mag_range[0], mag_range[1], num = mag_granularity)

    property_dict = {"Energy": {}, "Specific Heat": {}, "Magnetization": {}, "Susceptibility": {}, "Free Energy": {}}

    property_dict["Energy"][graph_id] = eig_to_property_3d(eigenvalues, mag, eigenvalues, mag_grid, temp_grid, number_sites)
    property_dict["Specific Heat"][graph_id] = (eig_to_property_3d(eigenvalues ** 2, mag, eigenvalues, mag_grid, temp_grid, number_sites) - (property_dict["Energy"][graph_id] ** 2)) / (temp_grid ** 2)
    property_dict["Magnetization"][graph_id] = eig_to_property_3d(mag, mag, eigenvalues, mag_grid, temp_grid, number_sites)
    property_dict["Susceptibility"][graph_id] = (eig_to_property_3d(mag ** 2, mag, eigenvalues, mag_grid, temp_grid, number_sites) - (property_dict["Magnetization"][graph_id] ** 2)) / (temp_grid)
    property_dict["Free Energy"][graph_id] = free_energy_3d(mag, eigenvalues, mag_grid, temp_grid)

    return(property_dict)

def ising_all(input_dict, graph_bond_info_ordered):
    """
    Takes in the graph bond information and returns
    the property dictionary unordered
    """
    benchmarking = input_dict["benchmarking"]

    property_list = []

    for order, graph_bond_info in graph_bond_info_ordered.items():
        property_list.extend([find_ising_properties(input_dict, graph_id, bond_info, order) for graph_id, bond_info in graph_bond_info.items()])

    if benchmarking:
        print("Starting Eigenvalue and Property Solving, this will take the longest time")
        pbar = ProgressBar()
        pbar.register()

    property_list_solved = dask.compute(property_list, scheduler = "processes", num_workers=input_dict["number_cores"])[0]

    if benchmarking:
        print("Finished Eigenvalue and Property Solving")

    property_dict = {"Energy": {}, "Specific Heat": {}, "Magnetization": {}, "Susceptibility": {}, "Free Energy": {}}
    for cluster in property_list_solved:
        for property_name, property_value in cluster.items():
            property_dict[property_name].update(property_value)

    return(property_dict)

@dask.delayed
def find_heisenberg_eigenvalues(bond_information, number_sites, tunneling_strength):
    """
    This function takes the bond information for a specific graph
    and returns its heisenberg energy eigenvalues
    """
    number_states = 2 ** number_sites
    hamiltonian_matrix = np.zeros((number_states, number_states))
    mag = []

    h = 0
    for state in range(number_states):
        for bond in bond_information:
            site1, site2, bond_strength = get_bit(state, bond[0]), get_bit(state, bond[1]), tunneling_strength[bond[2] - 1]
            if site1 == site2:
                hamiltonian_matrix[state, state] += 0.25 * bond_strength
            else:
                hamiltonian_matrix[state, state] -= 0.25 * bond_strength
                new_state = flip_bit(flip_bit(state, bond[0]), bond[1])
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
        eig_dict.update({k: find_heisenberg_eigenvalues(v, order, tunneling_strength) for k, v in graph_bond_info.items()})

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
