import ExactDiag
import numpy as np
import scipy
import json

order = int(input("Order Number for square?: "))
nlce_type = 'square'
temp_range = (-1, 2)
granularity = 100

def solve_energy_for_order(data_dir, order, nlce_type, temp_range, granularity):

    temperature_array = np.logspace(temp_range[0], temp_range[1], granularity)
    
    graph_property_info = {}
    
    graph_bond = open(f'{data_dir}/graph_bond_{nlce_type}_{order}.json')
    graph_bond_dict = json.load(graph_bond)
    
    # Parallellize here
    for graph_id in graph_bond_dict:
        bond_info = graph_bond_dict[graph_id]
        energies = ExactDiag.solve_energies(order, [], bond_info)
    
        exp_energy_temp_matrix = np.exp(-energies[:, np.newaxis] / temperature_array)
        partition_function = exp_energy_temp_matrix.sum(axis=0)
        energy_function = np.matmul(energies, exp_energy_temp_matrix)
        final_energies = energy_function / partition_function
        graph_property_info[graph_id] = list(final_energies)
    
    
    graph_property_info_json = open(f"{data_dir}/graph_energy_info_{order}_{nlce_type}.json", "w")
    json.dump(graph_property_info, graph_property_info_json)
    
    return(graph_property_info)

