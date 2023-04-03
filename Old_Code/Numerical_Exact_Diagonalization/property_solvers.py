#!/usr/bin/env python3
import numpy as np
import scipy
import json
import multiprocessing as mp
from functools import partial
import pathlib

import exact_diagonalization as ed
from model_specific_functions import *

def energy_solver(graph_id, model, order, graph_bond_dict, temperature_array):
    graph_property_info = {}
    bond_info = graph_bond_dict[graph_id]
    energies = ed.solve_for_property(model_info[model]["generator"], model_info[model]["bond_solver"], model_info[model]["site_solver"], order, [], bond_info)

    exp_energy_temp_matrix = np.exp(-energies[:, np.newaxis] / temperature_array)
    partition_function = exp_energy_temp_matrix.sum(axis=0)
    energy = np.matmul(energies, exp_energy_temp_matrix)

    final_energies = energy / partition_function

    graph_property_info[graph_id] = list(final_energies)

    return(graph_property_info)

def specific_heat_solver(graph_id, model, order, graph_bond_dict, temperature_array):
    graph_property_info = {}
    bond_info = graph_bond_dict[graph_id]
    energies = ed.solve_for_property(model_info[model]["generator"], model_info[model]["bond_solver"], model_info[model]["site_solver"], order, [], bond_info)

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


def solve_property_for_order(property_type, model_name, nlce_data_dir, property_data_dir, order, nlce_type, temperature_array, benchmarking = False):
    graph_property_path = f"{property_data_dir}/graph_{model_name}_{property_type}_info_{nlce_type}_{order}.json"
    graph_bond_info_path = f"{nlce_data_dir}/{nlce_type}/graph_bond_{nlce_type}_{order}.json"

    if (not pathlib.Path(graph_property_path).exists()) or benchmarking:
        graph_bond = open(graph_bond_info_path)
        graph_bond_dict = json.load(graph_bond)

        property_solver = partial(property_information[property_type], model = model_name, order = order, graph_bond_dict = graph_bond_dict, temperature_array = temperature_array)

        #cpus = mp.cpu_count()
        cpus = 8
        if benchmarking:
            print(f"Using {cpus} cpus")
        pool = mp.Pool(cpus)
        graph_property_list = list(pool.map(property_solver, graph_bond_dict.keys()))
        graph_property_info = {graph_id: graph_info for graph_dict in graph_property_list for graph_id, graph_info in graph_dict.items()}

        if not benchmarking:
            graph_property_info_json = open(graph_property_path, "w")
            json.dump(graph_property_info, graph_property_info_json)

    else:
        graph_property_info_json = open(graph_property_path)
        graph_property_info = json.load(graph_property_info_json)

    return(graph_property_info)

property_information = {
    "energy": energy_solver,
    "specific_heat": specific_heat_solver
}

