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
