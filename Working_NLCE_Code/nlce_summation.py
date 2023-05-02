import json, sys, time, copy, os, pickle
import numpy as np
import pandas as pd
import exact_diagonalization as ed
import matplotlib.pyplot as plt

def euler_resummation(input_dict, weighted_property_dict)

def sum_property(input_dict, property_dict, graph_mult_ordered, subgraph_mult_ordered):
    """
    This function takes in the dictionary (not seperated by order)
    of the property of every graph and returns nlce sums seperated by 
    order of the property as a function of temperature

    input_dict: {inputs from the json file}
    property_dict: {graph_id: [unweighted_property_numpy_array]}
    subgraph_mult_dicts: {order: {graph_id: {subgraph_id: multiplicity}}}
                    print(property_dict[subgraph_id])
    graph_mult_dicts: {order: {graph_id: multiplicity}}

    return: {order: [property along temperature grid]}
    """
    benchmarking = input_dict["benchmarking"]
    # initialize total_property with all zeros
    total_property = np.zeros_like(list(property_dict.values())[0])
    weighted_property_dict = {}
    # Loop over every order's graph multiplicity
    for order, graph_mult_dict in graph_mult_ordered.items():
        # Loop over every graph per order
        if benchmarking:
            print(f"Starting NLCE Sum for order {order} with {len(graph_mult_dict)} graphs")
            start = time.time()
        for graph_id, graph_mult in graph_mult_dict.items():
            # find the initial weight for the graph
            graph_weight = property_dict[graph_id]
            if order > 1:
                # Loop over all the subgraphs of the graph
                subgraph_mult_dict = subgraph_mult_ordered[order]
                for subgraph_id, subgraph_mult in subgraph_mult_dict[graph_id].items():
                    # And subtract their weights (according to the subgraph multiplicity)
                    graph_weight -= subgraph_mult * property_dict[subgraph_id]

            # update the total property with the property of the given graph
            total_property += graph_mult * graph_weight
        if benchmarking:
            print(f"Finishing NLCE Sum for order {order} in {time.time() - start :.4f}s")
        # update the weighted property dictionary for this order
        weighted_property_dict[order] = np.copy(total_property)

    return(weighted_property_dict)

def load_dict(input_dict):
    """
    Loads the graph bond info, graph multiplicity and subgraph multplicity
    based on the input_dict
    """
    benchmarking = input_dict["benchmarking"]
    nlce_data_dir = input_dict["nlce_data_dir"]
    geometry = input_dict["geometry"]
    graph_dict_types = (f"graph_bond_{geometry}", f"graph_mult_{geometry}", f"subgraph_mult_{geometry}")
    
    graph_dicts = [{}, {}, {}]
    for order in range(1, input_dict["final_order"] + 1):
        if benchmarking:
            start = time.time()
            print(f"Loading order {order} from json files")
        
        for ind, graph_dict_type in enumerate(graph_dict_types):

            graph_file = open(f'{nlce_data_dir}/{geometry}/{graph_dict_type}_{order}.json')
            graph_dict = json.load(graph_file)

            graph_dicts[ind][order] = graph_dict
        
        if benchmarking:
            print(f"Finished loading order {order} in {time.time() - start:.4f}s")

    return(graph_dicts)

def plot_property(input_dict, weight_dict, temp_grid, property_name):
    benchmarking = input_dict["benchmarking"]
    starting_order_plot = input_dict["starting_order_plot"]
    final_order = input_dict["final_order"]
    tunneling_strength = input_dict["tunneling_strength"]
    mag_range = input_dict["mag_range"]
    mag_granularity = input_dict["mag_granularity"]
    save_path = f"{input_dict['fig_output_dir']}/{input_dict['property']}_{property_name}_{input_dict['geometry']}_{final_order}.pdf"

    if benchmarking:
        print(f"Plotting {property_name}")

    for order in range(starting_order_plot, final_order + 1):
        
        plot_prop = weight_dict[order]
        if property_name == "Magnetization":
            mag_grid = np.linspace(mag_range[0], mag_range[1], num = mag_granularity)
            #plt.plot(mag_grid, plot_prop[:, -1], label = f"{order}")
            plt.plot(temp_grid, plot_prop[2, :], label = f"{order}")
        elif property_name == "Susceptibility":
            plt.plot(temp_grid, plot_prop[2, :], label = f"{order}")
        else:
            plt.plot(temp_grid, plot_prop, label = f"{order}")


    if property_name == "Specific Heat":
        plt.xlabel("Temperature") 
        plt.ylabel(f"{property_name}")
        plt.title(f"{property_name} vs Temperature for J = {tunneling_strength}") 
        plt.xscale('linear')
        plt.ylim([0, 1.2])
        plt.xlim([0, 3])

    elif property_name == "Entropy":
        plt.xlabel("Temperature") 
        plt.ylabel(f"{property_name}")
        plt.title(f"{property_name} vs Temperature for J = {tunneling_strength}") 
        plt.xscale('linear')
        plt.xlim([0, 5])
        plt.ylim([0, 0.7])
        plt.axhline(0, color='k')
        plt.axhline(np.log(2), color='k')
    
    elif property_name == "Energy":
        mcdata_ising = pd.read_csv("./Data/Monte_Carlo_Data/mcdata_ising.csv")
        temp_mc, e_mc = mcdata_ising['T'], mcdata_ising['E']/2500
        plt.plot(temp_mc, e_mc, 'k.',label = "MC Data")

        plt.xlabel("log(Temperature)") 
        plt.ylabel(f"{property_name}")
        plt.title(f"{property_name} vs Temperature for J = {tunneling_strength}") 
        plt.xscale('log')
        plt.ylim([-3, 0.5])
    
    elif property_name == "Magnetization":
        plt.xlabel("h") 
        plt.ylabel(f"{property_name}")
        plt.title(f"{property_name} vs h for J = {tunneling_strength}, T = 2") 
        plt.xscale('linear')

    elif property_name == "Susceptibility":
        plt.xlabel("log(Temperature)") 
        plt.ylabel(f"{property_name}")
        plt.title(f"{property_name} vs Temperature for J = {tunneling_strength}") 
        plt.xscale('log')
        plt.ylim([-0.5, 5])

    elif property_name == "Free Energy":
        plt.xlabel("log(Temperature)") 
        plt.ylabel(f"{property_name}")
        plt.title(f"{property_name} vs Temperature for J = {tunneling_strength}") 
        plt.xscale('log')
        plt.ylim([0, 1])

    plt.legend()
    plt.savefig(save_path)

    plt.close()
    if benchmarking:
        print(f"Finished Plotting {property_name}")

    return(save_path)

def main(input_dict):
    benchmarking = input_dict["benchmarking"]

    graph_bond_info_ordered, graph_mult_ordered, subgraph_mult_ordered = load_dict(input_dict)

    property_data_dir = f"{input_dict['output_dir']}/{input_dict['geometry']}/{input_dict['property']}/{input_dict['final_order']}"

    if input_dict["use_existing_data"]:
        if benchmarking:
            print(f"Loading existing data from {property_data_dir}")

        property_info = open(f"{property_data_dir}/property_info.pkl", 'rb')
        property_dict_all = pickle.load(property_info)
        temp_grid = property_dict_all.pop('temp_grid')

    else:
        if benchmarking:
            print("Computing New Data")
        property_dict_before_nlce_sum, temp_grid = ed.property_functions[input_dict["property"]](input_dict, graph_bond_info_ordered)

        property_dict_all = {}
        # This is a hacky solution, but it works since Energy sorts before Free Energy :)
        for prop_name, pre_summed in sorted(property_dict_before_nlce_sum.items()):
            property_dict_all[prop_name] = sum_property(input_dict, copy.deepcopy(pre_summed), graph_mult_ordered, subgraph_mult_ordered)
            if prop_name == "Free Energy":
                property_dict_all["Entropy"] = {order: ((property_dict_all["Free Energy"][order] + (avg_en / temp_grid))) for order, avg_en in copy.deepcopy(property_dict_all["Energy"]).items()}


        os.makedirs(property_data_dir, exist_ok=True)
        property_info = open(f"{property_data_dir}/property_info.pkl", 'wb')
        property_dict_all["temp_grid"] = temp_grid
        pickle.dump(property_dict_all, property_info)
        # Don't worry about it shhhh it works
        property_dict_all.pop("temp_grid")

    output_dirs = []
    for prop_name, summed_property in property_dict_all.items():
        
        output_dirs.append(plot_property(input_dict, 
                                         summed_property,
                                         temp_grid, 
                                         prop_name))

    return(output_dirs)

if __name__ == "__main__":
    input_dict = json.load(open(sys.argv[1]))
    output_dirs = main(input_dict)
    for output in output_dirs:
        print(f"Output File Saved in {output}")
