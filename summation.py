#!/usr/bin/env python3
import ExactDiag
import numpy as np
import json
import matplotlib.pyplot as plt
import copy
import time
import sys

start = time.time()
final_order = int(sys.argv[2])
temp_range = (.5, 3)
nlce_type = sys.argv[1]
granularity = 100
temp_grid = np.linspace(temp_range[0], temp_range[1], num=granularity)
weight_dict_ordered = {}
weight_dict = {}

for order in range(1, final_order + 1):
    #graph_property_info = ExactDiag.solve_specific_heat_for_order("./data", order, nlce_type, temp_grid)
    graph_property_info = ExactDiag.solve_energy_for_order("./data", order, nlce_type, temp_grid)

    subgraph_mult = open(f'./data/subgraph_mult_{nlce_type}_{order}.json')
    subgraph_mult_dict = json.load(subgraph_mult)

    weight_dict_ordered[order] = {}

    for graph_id in subgraph_mult_dict:
        property_unweighted = np.array(graph_property_info[graph_id])
        for subgraph_id in subgraph_mult_dict[graph_id]:

            property_unweighted -= subgraph_mult_dict[graph_id][subgraph_id] * weight_dict[subgraph_id]

        weight_dict_ordered[order][graph_id] = property_unweighted
        weight_dict[graph_id] = property_unweighted


final_energy_grid_by_order = []
final_energy_grid = np.zeros_like(temp_grid)

for order in range(1, final_order + 1):
    graph_mult = open(f'./data/graph_mult_{nlce_type}_{order}.json')
    graph_mult_dict = json.load(graph_mult)
    for graph_id in weight_dict_ordered[order]:
        final_energy_grid += graph_mult_dict[graph_id] * weight_dict[graph_id]

    final_energy_grid_by_order.append(copy.deepcopy(final_energy_grid))


for order in range(6, len(final_energy_grid_by_order)):
    plt.plot(temp_grid, final_energy_grid_by_order[order], label = f"{order + 1}")


plt.legend()

plt.savefig(f"/home/rightleftspin/Git_Repos/HubbardModel/data/energy_{final_order}.pdf")
print(f"Total Time: {time.time() - start: .4f}")
