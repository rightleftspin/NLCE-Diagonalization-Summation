import dask
import sys, json
import numpy as np
import pandas as pd
import exact_diagonalization as ed
import matplotlib.pyplot as plt

def energy_solver(cluster_eigenvalues, temperature_array):
    """
    This function takes in the cluster eigenvalues and transforms
    it into an energy array fit to be plotted with the input
    temperature array
    """
    exp_energy_temp_matrix = np.exp(-cluster_eigenvalues[:, np.newaxis] / temperature_array)
    partition_function = exp_energy_temp_matrix.sum(axis=0)
    energy = np.matmul(cluster_eigenvalues, exp_energy_temp_matrix)

    final_energies = energy / partition_function

    return(final_energies)

def find_cluster_total_property(cluster_id, cluster_property, subcluster_multiplicity, all_cluster_properties):
    """
    Finds the total nlce property of a cluster if all its underlying subclusters are taken
    into consideration through the subcluster_multiplicity
    """
    for subcluster_id, multiplicity in subcluster_multiplicity.items():
        cluster_property -= multiplicity * all_cluster_properties[subcluster_id]

    return(cluster_property)

def find_properties_for_order(order, cluster_bond_info, model, tunneling_strength, temperature_array):
    """
    Finds the property evaluated at each cluster for the given model
    """
    all_cluster_eigenvalues_initial = ed.ed_main(order, cluster_bond_info, model, tunneling_strength)
    all_cluster_eigenvalues = dask.compute(all_cluster_eigenvalues_initial, scheduler = "processes")[0]
    property_solver = property_solvers[model]

    cluster_properties = {}

    for cluster_id, cluster_eigenvalues in all_cluster_eigenvalues.items():
        cluster_properties[cluster_id] = property_solver(cluster_eigenvalues, temperature_array)

    return(cluster_properties)

def update_properties_with_subclusters(cluster_properties, all_cluster_properties, subcluster_multiplicity):
    """
    Takes a dictionary of clusters and updates the cluster property for the total NLCE
    sum with the subclusters considered
    """
    cluster_properties_updated = {}
    for cluster_id, cluster_property in cluster_properties.items():
        cluster_properties_updated[cluster_id] = find_cluster_total_property(cluster_id, cluster_property, subcluster_multiplicity[cluster_id], all_cluster_properties)

    return(cluster_properties_updated)

def sum_all_orders(clusters_by_order, all_cluster_properties, cluster_mult_all_orders):
    """
    Takes cluster_ids seperated by order and returns an array seperated
    by order, with each order considering the last in the calculation
    """
    seperated_by_order = []
    running_cluster_property = np.zeros_like(list(all_cluster_properties.values())[0])
    for order, cluster_ids in clusters_by_order.items():
        for cluster_id in cluster_ids:
            running_cluster_property += cluster_mult_all_orders[order][cluster_id] * all_cluster_properties[cluster_id]

        seperated_by_order.append(running_cluster_property.copy())

    return(seperated_by_order)

def load_json_all_orders(nlce_data_dir, final_order, nlce_type):
    """
    Takes in the data directory and final order and loads all the
    dictionaries of each input file into their corresponding spots
    """
    subgraph_mult_all_orders, graph_mult_all_orders, graph_bond_all_orders = {}, {}, {}

    clusters_by_order = {}

    for order in range(1, final_order + 1):
        subgraph_mult_file = open(f'{nlce_data_dir}/{nlce_type}/subgraph_mult_{nlce_type}_{order}.json')
        graph_mult_file = open(f'{nlce_data_dir}/{nlce_type}/graph_mult_{nlce_type}_{order}.json')
        graph_bond_file = open(f'{nlce_data_dir}/{nlce_type}/graph_bond_{nlce_type}_{order}.json')

        graph_mult = json.load(graph_mult_file)
        graph_mult_all_orders[order] = graph_mult
        subgraph_mult_all_orders[order] = json.load(subgraph_mult_file)
        graph_bond_all_orders[order] = json.load(graph_bond_file)

        clusters_by_order[order] = list(graph_mult.keys())

    return(graph_mult_all_orders, subgraph_mult_all_orders, graph_bond_all_orders, clusters_by_order)

def main(final_order, nlce_data_dir, output_dir, nlce_type, model, tunneling_strength, temperature_array):
    """
    Main function runs the whole NLCE sum for all orders
    """

    # Load all json data
    graph_mult_all_orders, subgraph_mult_all_orders, graph_bond_all_orders, clusters_by_order = load_json_all_orders(nlce_data_dir, final_order, nlce_type)

    cluster_properties_by_order = {}
    all_cluster_properties = {}
    # Find all the initial properties for each order
    for order, cluster_bond_info in graph_bond_all_orders.items():
        cluster_properties = find_properties_for_order(order, cluster_bond_info, model, tunneling_strength, temperature_array)
        if order > 1:
            updated_properties = update_properties_with_subclusters(cluster_properties, all_cluster_properties, subgraph_mult_all_orders[order])
            cluster_properties_by_order[order] = updated_properties
            all_cluster_properties.update(updated_properties)
        else:
            cluster_properties_by_order[order] = cluster_properties
            all_cluster_properties.update(cluster_properties)


    property_sep_by_order = sum_all_orders(cluster_properties_by_order, all_cluster_properties, graph_mult_all_orders)


    for order in range(6, final_order):
        plt.plot(temperature_array, property_sep_by_order[order], label = f"{order + 1}")


    mcdata_ising = pd.read_csv("./mcdata_ising.csv")
    temp_mc, e_mc = mcdata_ising['T'], mcdata_ising['E']/2500
    plt.plot(temp_mc, e_mc, 'k.',label = "MC Data")
    plt.xlabel("Log(Temperature)")
    plt.ylabel(f"{model.replace('_', ' ').capitalize()}")
    plt.title(f"{model.replace('_', ' ')} vs Log(Temperature) NLCE Order {final_order} {nlce_type}")
    plt.legend()
    plt.xscale("log")
    plt.xlim([.1, 20])

    plt.savefig(f"{output_dir}/{model}_{final_order}.pdf")
    plt.close()

    return()

property_solvers = { "heisenberg_energy": energy_solver, "ising_energy": energy_solver }

if __name__ == "__main__":
    final_order = eval(sys.argv[1])
    nlce_data_dir = sys.argv[2]
    nlce_type = sys.argv[3]
    model = sys.argv[4]
    tunneling_strength = [1]
    temperature_array = np.linspace(.1, 20, 50)
    output_dir = "./Data/Plots"
    main(final_order, nlce_data_dir, output_dir, nlce_type, model, tunneling_strength, temperature_array)

