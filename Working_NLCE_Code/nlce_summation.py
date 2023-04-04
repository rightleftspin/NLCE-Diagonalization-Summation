import json, sys, time
import numpy as np
import pandas as pd
import exact_diagonalization as ed
import matplotlib.pyplot as plt

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
    total_property = np.zeros(input_dict["grid_granularity"])
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
    save_path = f"{input_dict['fig_output_dir']}/{input_dict['property']}_{property_name}_{input_dict['geometry']}_{final_order}.pdf"

    if benchmarking:
        print(f"Plotting {property_name}")

    #plt.figure()
    for order in range(starting_order_plot, final_order + 1):
        
        plot_prop = weight_dict[order]
        plt.plot(temp_grid, plot_prop, label = f"{order}")

        plt.xlabel("log(Temperature)") 
        plt.ylabel(f"{property_name}")
        plt.title(f"{property_name} vs Temperature for J = {tunneling_strength}") 
        plt.xscale('log')

#    mcdata_ising = pd.read_csv("./mcdata_ising.csv")
#    temp_mc, e_mc = mcdata_ising['T'], mcdata_ising['E']/2500
#    plt.plot(temp_mc, e_mc, 'k.',label = "MC Data")

    if property_name == "Specific Heat":
        plt.ylim([-0.5, 3])

    elif property_name == "Entropy":
        plt.ylim([0, 1.5])
        plt.axhline(0, color='k')
        plt.axhline(np.log(2), color='k')
    
    elif property_name == "Energy":
        plt.ylim([-6, 0.5])
    
    elif property_name == "Magnetization":
        plt.ylim([-0.5, 5])

    elif property_name == "Susceptibility":
        plt.ylim([-0.5, 5])

    plt.legend()
    plt.savefig(save_path)

    plt.close()
    if benchmarking:
        print(f"Finished Plotting {property_name}")

    return(save_path)

def main(input_dict):

    graph_bond_info_ordered, graph_mult_ordered, subgraph_mult_ordered = load_dict(input_dict)
    
    property_dict_all, temp_grid = ed.property_functions[input_dict["property"]](input_dict, graph_bond_info_ordered)

    output_dirs = []
    for prop_name, property_dict in property_dict_all.items():
        if prop_name == "Free Energy":
            free_energy = sum_property(input_dict, property_dict, graph_mult_ordered, subgraph_mult_ordered)
            entropy = {order: ((free_energy[order] + (avg_en / temp_grid))) for order, avg_en in sum_property(input_dict, property_dict_all["Energy"], graph_mult_ordered, subgraph_mult_ordered).items()}
            output_dirs.append(plot_property(input_dict, 
                                         entropy,
                                         temp_grid, 
                                         "Entropy"))
        else:
            output_dirs.append(plot_property(input_dict, 
                                         sum_property(input_dict, property_dict, graph_mult_ordered, subgraph_mult_ordered),
                                         temp_grid, 
                                         prop_name))

    return(output_dirs)

if __name__ == "__main__":
    input_dict = json.load(open(sys.argv[1]))
    output_dirs = main(input_dict)
    for output in output_dirs:
        print(f"Output File Saved in {output}")
