import json, sys, time, copy, os, pickle, math
import numpy as np
import pandas as pd
import exact_diagonalization as ed
import matplotlib
import matplotlib.pyplot as plt

def euler_resummation(input_dict, weighted_property_dict):
    """
    This function takes in the weighted property dictionary
    and returns the resummed property using the euler method
    starting at the resummation order
    """
    benchmarking = input_dict["benchmarking"]

    shape_prop = weighted_property_dict[1].shape
    size_prop = weighted_property_dict[1].size
    big = np.float64(1e15 * size_prop)
    small = np.float64(1e-15 * size_prop)
    eps = 0.1 * size_prop

    e_loop = {k:weighted_property_dict[k] for k in range(5, max(weighted_property_dict.keys()) + 1) if k in weighted_property_dict}
    e = copy.deepcopy(weighted_property_dict)
    e[0] = np.copy(weighted_property_dict[5])
    n = 5
    ncv = 0
    val = np.zeros(shape_prop)
    lastval = np.copy(weighted_property_dict[5])
    for order, property_sum in sorted(e_loop.items()):
        e[order] = copy.deepcopy(property_sum)
        temp2 = np.zeros(shape_prop)
        for j in range(n, 0, -1):
            temp1 = temp2.copy()
            temp2 = e[j - 1]
            diff = e[j] - temp2
            if np.sum(np.abs(diff)) < small:
                e[j - 1] = big * np.ones(shape_prop)
            else:
                e[j - 1] = temp1 + (1 / diff)

        n += 1
        if (order % 2) == 0:
            val = np.copy(e[0])
        else:
            val = np.copy(e[1])
        if (np.sum(np.abs(val))) > (0.01 * big):
            val = lastval
        lasteps = np.sum(np.abs(val - lastval))
        if lasteps > eps:
            ncv = 0
        else:
            ncv += 1

# Comment out to check if the series is converging to some extent
#        if ncv > 3:
#            if benchmarking:
#                print("Series Converged to Some Reasonable Extent")

        lastval = val

    return(val)

def eps_wynn(k, n, weighted_property_dict):
    if k == 0:
        return(weighted_property_dict[n])
    elif k == -1:
        return(np.zeros_like(weighted_property_dict[n]))
    else:
        first = eps_wynn(k - 2, n + 1, weighted_property_dict)
        second = eps_wynn(k - 1, n + 1, weighted_property_dict) - eps_wynn(k - 1, n, weighted_property_dict)
        total = first + 1 / second
        return(total)

def wynn_resummation(input_dict, weighted_property_dict):
    """
    This function takes in the weighted property dictionary
    and returns the resummed property using the euler method
    starting at the resummation order
    """
    benchmarking = input_dict["benchmarking"]
    final_order = input_dict["final_order"]
    wynn_cycles = input_dict["wynn_cycles"]

    wynn_correction = eps_wynn((2 * wynn_cycles), final_order - (2 * wynn_cycles), copy.deepcopy(weighted_property_dict))

    return(wynn_correction)

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
                    graph_weight -= subgraph_mult[1] * property_dict[subgraph_id]

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

def plot_property(input_dict, weight_dict, property_name):
    benchmarking = input_dict["benchmarking"]
    starting_order_plot = input_dict["starting_order_plot"]
    final_order = input_dict["final_order"]
    tunneling_strength = input_dict["tunneling_strength"]
    temp_range = input_dict["temp_range"]
    grid_granularity = input_dict["grid_granularity"]
    mag_range = input_dict["mag_range"]
    mag_granularity = input_dict["mag_granularity"]

    save_path = f"{input_dict['fig_output_dir']}/{input_dict['property']}_{property_name}_{input_dict['geometry']}_{final_order}.pdf"

    temp_grid = np.logspace(temp_range[0], temp_range[1], num = grid_granularity)
    mag_grid = np.linspace(mag_range[0], mag_range[1], num = mag_granularity)

    if benchmarking:
        print(f"Plotting {property_name}")

    if property_name == "Specific Heat":
        plt.figure()

        magnetization_index = 0
        plt.plot(temp_grid, weight_dict[final_order - 0][magnetization_index, :], label = f"Order {final_order - 0}")
        plt.plot(temp_grid, weight_dict[final_order - 1][magnetization_index, :], label = f"Order {final_order - 1}")
        plt.plot(temp_grid, weight_dict[final_order - 2][magnetization_index, :], label = f"Order {final_order - 2}")
        plt.plot(temp_grid, weight_dict[final_order - 3][magnetization_index, :], label = f"Order {final_order - 3}")

        t = [0.75, 0.82692308, 0.90384615, 0.98076923, 1.05769231, 1.13461538, 1.21153846, 1.28846154, 1.36538462, 1.44230769, 1.51923077, 1.59615385, 1.67307692, 1.75, 1.82692308, 1.90384615, 1.98076923, 2.05769231, 2.13461538, 2.21153846, 2.28846154, 2.36538462, 2.44230769, 2.51923077, 2.59615385, 2.67307692, 2.75, 2.82692308, 2.90384615, 2.98076923, 3.05769231, 3.13461538, 3.21153846, 3.28846154, 3.36538462, 3.44230769, 3.51923077, 3.59615385, 3.67307692, 3.75]

        cv = [8.92246301e0, 7.25267436e-3, 6.73706106e-02, 2.12648464e-02, 3.49100949e-02, 4.95167591e-02, 7.10201532e-02, 9.34417527e-02, 1.24343499e-01, 1.59397921e-01, 2.12261711e-01, 2.63698043e-01, 3.18925557e-01, 3.79358295e-01, 4.67694032e-01, 5.66367144e-01, 6.89486655e-01, 9.08743234e-01, 1.07833132e+00, 1.19233267e+00, 1.99714932e+00, 1.37233900e+00, 9.76795423e-01, 8.22428363e-01, 7.35767311e-01, 6.11553886e-01, 5.57913038e-01, 4.77231568e-01, 4.60485195e-01, 4.23743619e-01, 3.75628841e-01, 3.49760489e-01, 3.23839163e-01, 2.98469016e-01, 2.83140165e-01, 2.63471189e-01, 2.45627934e-01, 2.26596287e-01, 2.18642792e-01, 2.01324247e-01]

        plt.plot(np.array(t)/4, np.array(cv), 'k.', label = f"MC 40x40")
        #plt.plot(temp_grid, wynn_resummation(input_dict, weight_dict)[magnetization_index, :], 'c', label = f"{final_order} sites (Wynn)")

        #weight_dict_min_one = {k:weight_dict[k] for k in range(final_order) if k in weight_dict}

        #plt.plot(temp_grid, euler_resummation(input_dict, weight_dict)[magnetization_index, :], 'b-', label = f"{final_order} sites (Euler)")
        #plt.plot(temp_grid, euler_resummation(input_dict, weight_dict_min_one)[magnetization_index, :], 'b--', label = f"{final_order - 1} sites (Euler)")
        plt.xlabel("T") 
        plt.ylabel(f"$C_v$")
        plt.title(f"{property_name} vs Temperature for J = 1, $k_b$ = 1")
        plt.xscale('log')
        plt.ylim([0, 1.1])
        plt.xlim([0.2, 5])

    elif property_name == "Entropy":
        plt.figure()

        magnetization_index = 0

        plt.plot(temp_grid, weight_dict[final_order - 0][magnetization_index, :], label = f"Order {final_order - 0}")
        plt.plot(temp_grid, weight_dict[final_order - 1][magnetization_index, :], label = f"Order {final_order - 1}")
        plt.plot(temp_grid, weight_dict[final_order - 2][magnetization_index, :], label = f"Order {final_order - 2}")
        plt.plot(temp_grid, weight_dict[final_order - 3][magnetization_index, :], label = f"Order {final_order - 3}")
        #plt.plot(temp_grid, wynn_resummation(input_dict, weight_dict)[magnetization_index, :], 'c', label = f"{final_order} sites (Wynn)")

        #weight_dict_min_one = {k:weight_dict[k] for k in range(final_order) if k in weight_dict}

        #plt.plot(temp_grid, euler_resummation(input_dict, weight_dict)[magnetization_index, :], 'b-', label = f"{final_order} sites (Euler)")
        #plt.plot(temp_grid, euler_resummation(input_dict, weight_dict_min_one)[magnetization_index, :], 'b--', label = f"{final_order - 1} sites (Euler)")

        plt.xlabel("T")
        plt.ylabel(f"S")
        plt.title(f"{property_name} vs Temperature for J = 1, $k_b$ = 1")
        plt.xscale('log')
        plt.ylim([0, 0.8])
        plt.xlim([0.35, 2.5])

    elif property_name == "Energy":
        plt.figure()
        magnetization_index = 0

       # mcdata_ising = pd.read_csv("./Data/Monte_Carlo_Data/mcdata_ising.csv")
       # temp_mc, e_mc = mcdata_ising['T']/4, mcdata_ising['E']/(2500 * 4)
       # plt.plot(temp_mc, e_mc, 'k--',label = "MC Data")

        plt.plot(temp_grid, weight_dict[final_order - 0][magnetization_index, :], label = f"Order {final_order - 0}")
        plt.plot(temp_grid, weight_dict[final_order - 1][magnetization_index, :], label = f"Order {final_order - 1}")
        plt.plot(temp_grid, weight_dict[final_order - 2][magnetization_index, :], label = f"Order {final_order - 2}")
        plt.plot(temp_grid, weight_dict[final_order - 3][magnetization_index, :], label = f"Order {final_order - 3}")
        #plt.plot(temp_grid, weight_dict[final_order - 3][magnetization_index, :], 'y', label = f"{final_order - 3} sites")
        #plt.plot(temp_grid, weight_dict[final_order - 4][magnetization_index, :], 'k', label = f"{final_order - 4} sites")
        #plt.plot(temp_grid, wynn_resummation(input_dict, weight_dict)[magnetization_index, :], 'c', label = f"{final_order} sites (Wynn)")

        #weight_dict_min_one = {k:weight_dict[k] for k in range(final_order) if k in weight_dict}

        #plt.plot(temp_grid, euler_resummation(input_dict, weight_dict)[magnetization_index, :], 'b-', label = f"{final_order} sites (Euler)")
        #plt.plot(temp_grid, euler_resummation(input_dict, weight_dict_min_one)[magnetization_index, :], 'b--', label = f"{final_order - 1} sites (Euler)")

        t = [0.75, 0.82692308, 0.90384615, 0.98076923, 1.05769231, 1.13461538, 1.21153846, 1.28846154, 1.36538462, 1.44230769, 1.51923077, 1.59615385, 1.67307692, 1.75, 1.82692308, 1.90384615, 1.98076923, 2.05769231, 2.13461538, 2.21153846, 2.28846154, 2.36538462, 2.44230769, 2.51923077, 2.59615385, 2.67307692, 2.75, 2.82692308, 2.90384615, 2.98076923, 3.05769231, 3.13461538, 3.21153846, 3.28846154, 3.36538462, 3.44230769, 3.51923077, 3.59615385, 3.67307692, 3.75]
        en = [-1.946043, -1.9993945, -1.87842175, -1.99748975, -1.995458, -1.992419, -1.98774725, -1.9814375, -1.972771, -1.96142, -1.94711775, -1.9294155, -1.9069245, -1.87933375, -1.84643025, -1.80704, -1.7590435, -1.69804625, -1.62947025, -1.5402415, -1.39872875, -1.25698075, -1.1578565, -1.0918195, -1.03353, -0.98049775, -0.93451675, -0.89232725, -0.85794675, -0.824806, -0.79534625, -0.76530375, -0.74140325, -0.71692, -0.69659575, -0.67303375, -0.654429, -0.6366435, -0.62011475, -0.60565325]

        plt.plot(np.array(t)/4, np.array(en)/4, 'k.', label = f"MC 40x40")
        plt.xlabel("T")
        plt.ylabel(f"E")
        plt.title(f"{property_name} vs Temperature for J = 1, $k_b$ = 1")
        plt.xscale('log')
        plt.ylim([-0.6, 0])
        plt.xlim([0.26, 1.8])

    elif property_name == "Magnetization":
        plt.figure()

        """
            Temperature Plots for given Magnetization
        """
        magnetization_indices = np.linspace(2, mag_grid.size - 10, num = 10, dtype = int)

        #plt.plot(temp_grid, weight_dict[final_order][magnetization_index, :], 'r-', label = f"{final_order} sites")
        #plt.plot(temp_grid, weight_dict[final_order - 1][magnetization_index, :], 'r--', label = f"{final_order - 1} sites")
        for magnetization_index in magnetization_indices:
            plt.plot(temp_grid, wynn_resummation(input_dict, weight_dict)[magnetization_index, :], label = f"h = {mag_grid[magnetization_index]:.2f}")

        #weight_dict_min_one = {k:weight_dict[k] for k in range(final_order) if k in weight_dict}

        #plt.plot(temp_grid, euler_resummation(input_dict, weight_dict)[magnetization_index, :], 'b-', label = f"{final_order} sites (Euler)")
        #plt.plot(temp_grid, euler_resummation(input_dict, weight_dict_min_one)[magnetization_index, :], 'b--', label = f"{final_order - 1} sites (Euler)")

        plt.xlabel("Temperature")
        plt.ylabel(f"{property_name}")
        plt.title(f"{property_name} vs Temperature for J = {tunneling_strength[0]}")
        plt.xscale('log')
        plt.ylim([0, 1.1])
        plt.xlim([0.2, 8])

        """
            Magnetization plots for given temperature
        """
        #temp_index = 0

        #plt.plot(mag_grid, weight_dict[final_order][:, temp_index], 'r-', label = f"{final_order} sites")
        #plt.plot(mag_grid, weight_dict[final_order - 1][:, temp_index], 'r--', label = f"{final_order - 1} sites")
        #plt.plot(mag_grid, wynn_resummation(input_dict, weight_dict)[:, temp_index], 'c', label = f"{final_order} sites (Wynn)")

        #weight_dict_min_one = {k:weight_dict[k] for k in range(final_order) if k in weight_dict}

        #plt.plot(mag_grid, euler_resummation(input_dict, weight_dict)[:, temp_index], 'b-', label = f"{final_order} sites (Euler)")
        #plt.plot(mag_grid, euler_resummation(input_dict, weight_dict_min_one)[:, temp_index], 'b--', label = f"{final_order - 1} sites (Euler)")

        #plt.xlabel("h")
        #plt.ylabel(f"{property_name}")
        #plt.title(f"{property_name} vs h for J = {tunneling_strength[0]} at T = {temp_grid[temp_index]:.3f}")
        #plt.xscale('linear')
        #plt.xlim([0, 6])
        #plt.ylim([0, 0.6])

    elif property_name == "Susceptibility":
        plt.figure()
        """
            Temperature Plots for given Magnetization
        """
        #magnetization_index = 0

        #plt.plot(temp_grid, weight_dict[final_order][magnetization_index, :], 'r-', label = f"{final_order} sites")
        #plt.plot(temp_grid, weight_dict[final_order - 1][magnetization_index, :], 'r--', label = f"{final_order - 1} sites")
        #plt.plot(temp_grid, wynn_resummation(input_dict, weight_dict)[magnetization_index, :], 'c', label = f"{final_order} sites (Wynn)")

        #weight_dict_min_one = {k:weight_dict[k] for k in range(final_order) if k in weight_dict}

        #plt.plot(temp_grid, euler_resummation(input_dict, weight_dict)[magnetization_index, :], 'b-', label = f"{final_order} sites (Euler)")
        #plt.plot(temp_grid, euler_resummation(input_dict, weight_dict_min_one)[magnetization_index, :], 'b--', label = f"{final_order - 1} sites (Euler)")

        #plt.xlabel("Temperature")
        #plt.ylabel(f"{property_name}")
        #plt.title(f"{property_name} vs Temperature for J = {tunneling_strength[0]} at h = {mag_grid[magnetization_index]:.3f}")
        #plt.xscale('log')
        #plt.ylim([0, 1])
        """
            Magnetization plots for given temperature
        """
        temp_index = 0

        plt.plot(mag_grid, weight_dict[final_order][:, temp_index], 'r-', label = f"{final_order} sites")
        plt.plot(mag_grid, weight_dict[final_order - 1][:, temp_index], 'r--', label = f"{final_order - 1} sites")
        plt.plot(mag_grid, wynn_resummation(input_dict, weight_dict)[:, temp_index], 'c', label = f"{final_order} sites (Wynn)")

        weight_dict_min_one = {k:weight_dict[k] for k in range(final_order) if k in weight_dict}

        plt.plot(mag_grid, euler_resummation(input_dict, weight_dict)[:, temp_index], 'b-', label = f"{final_order} sites (Euler)")
        plt.plot(mag_grid, euler_resummation(input_dict, weight_dict_min_one)[:, temp_index], 'b--', label = f"{final_order - 1} sites (Euler)")

        plt.xlabel("h")
        plt.ylabel(f"{property_name}")
        plt.title(f"{property_name} vs h for J = {tunneling_strength[0]} at T = {temp_grid[temp_index]:.3f}")
        plt.xscale('linear')
        plt.xlim([0, 6])
        plt.ylim([0, 0.6])

    elif property_name == "Free Energy":
        plt.figure()

        magnetization_index = 0

        plt.plot(temp_grid, weight_dict[final_order][magnetization_index, :], 'r-', label = f"{final_order} sites")
        plt.plot(temp_grid, weight_dict[final_order - 1][magnetization_index, :], 'r--', label = f"{final_order - 1} sites")
        plt.plot(temp_grid, wynn_resummation(input_dict, weight_dict)[magnetization_index, :], 'c', label = f"{final_order} sites (Wynn)")

        weight_dict_min_one = {k:weight_dict[k] for k in range(final_order) if k in weight_dict}

        plt.plot(temp_grid, euler_resummation(input_dict, weight_dict)[magnetization_index, :], 'b-', label = f"{final_order} sites (Euler)")
        plt.plot(temp_grid, euler_resummation(input_dict, weight_dict_min_one)[magnetization_index, :], 'b--', label = f"{final_order - 1} sites (Euler)")

        plt.xlabel("Temperature")
        plt.ylabel(f"{property_name}")
        plt.title(f"{property_name} vs Temperature for J = {tunneling_strength[0]} at h = {mag_grid[magnetization_index]:.3f}")
        plt.xscale('log')
        plt.ylim([-0.4, 0])
        plt.xlim([0.01, 3])

    plt.legend()
    plt.savefig(save_path)

    plt.close()
    if benchmarking:
        print(f"Finished Plotting {property_name}")

    return(save_path)

def main(input_dict):

    temp_range = input_dict["temp_range"]
    grid_granularity = input_dict["grid_granularity"]
    benchmarking = input_dict["benchmarking"]

    graph_bond_info_ordered, graph_mult_ordered, subgraph_mult_ordered = load_dict(input_dict)

    property_data_dir = f"{input_dict['output_dir']}/{input_dict['geometry']}/{input_dict['property']}/{input_dict['final_order']}"
    temp_grid = np.logspace(temp_range[0], temp_range[1], num = grid_granularity)

    if input_dict["use_existing_data"]:
        if benchmarking:
            print(f"Loading existing data from {property_data_dir}")

        property_info = open(f"{property_data_dir}/property_info.pkl", 'rb')
        property_dict_all = pickle.load(property_info)
        #temp_grid = property_dict_all.pop('temp_grid')

    else:
        if benchmarking:
            print("Computing New Data")
        property_dict_before_nlce_sum = ed.property_functions[input_dict["property"]](input_dict, graph_bond_info_ordered)

        property_dict_all = {}
        # This is a hacky solution, but it works since Energy sorts before Free Energy :)
        for prop_name, pre_summed in sorted(property_dict_before_nlce_sum.items()):
            property_dict_all[prop_name] = sum_property(input_dict, copy.deepcopy(pre_summed), graph_mult_ordered, subgraph_mult_ordered)
            if prop_name == "Free Energy":
                property_dict_all["Entropy"] = {order: ((property_dict_all["Free Energy"][order] + (avg_en / temp_grid))) for order, avg_en in copy.deepcopy(property_dict_all["Energy"]).items()}


        os.makedirs(property_data_dir, exist_ok=True)
        property_info = open(f"{property_data_dir}/property_info.pkl", 'wb')
        pickle.dump(property_dict_all, property_info)

    output_dirs = []
    for prop_name, summed_property in property_dict_all.items():
        output_dirs.append(plot_property(input_dict, 
                                         summed_property,
                                         prop_name))

    return(output_dirs)

if __name__ == "__main__":
    font = {'size': 13}

    matplotlib.rc('font', **font)
    input_dict = json.load(open(sys.argv[1]))
    output_dirs = main(input_dict)
    for output in output_dirs:
        print(f"Output File Saved in {output}")
