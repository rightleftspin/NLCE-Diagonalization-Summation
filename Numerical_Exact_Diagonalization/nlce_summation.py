#!/usr/bin/env python3
import numpy as np
import json
import matplotlib.pyplot as plt
import copy
import time
import sys
import os

import exact_diagonalization as ed
import property_solvers as ps
from model_specific_functions import model_info

start = time.time()
nlce_type = sys.argv[1]
final_order = int(sys.argv[2])
model = sys.argv[3]
property_type = sys.argv[4]
if (model not in model_info) or (property_type not in ps.property_information):
    raise ValueError("Not a valid Model or Property Type")

temp_range = (.5, 20)
granularity = 500
nlce_data_dir = "../Data/NLCE_Data"
property_data_dir = "../Data/Property_Data"
output_dir = "./output"
proper_property_data_dir = f"{property_data_dir}/{nlce_type}/{model}/{property_type}/{int(temp_range[0]*10)}_{int(temp_range[1]*10)}/"
os.makedirs(proper_property_data_dir, exist_ok=True)

temp_grid = np.linspace(temp_range[0], temp_range[1], num=granularity)
weight_dict_ordered = {}
weight_dict = {}

for order in range(1, final_order + 1):
    graph_property_info = ps.solve_property_for_order(property_type, model, nlce_data_dir, proper_property_data_dir, order, nlce_type, temp_grid)

    subgraph_mult = open(f'{nlce_data_dir}/subgraph_mult_{nlce_type}_{order}.json')
    subgraph_mult_dict = json.load(subgraph_mult)

    weight_dict_ordered[order] = {}

    for graph_id in subgraph_mult_dict:
        property_unweighted = np.array(graph_property_info[graph_id])
        for subgraph_id in subgraph_mult_dict[graph_id]:

            property_unweighted -= subgraph_mult_dict[graph_id][subgraph_id] * weight_dict[subgraph_id]

        weight_dict_ordered[order][graph_id] = property_unweighted
        weight_dict[graph_id] = property_unweighted

    print(f"Finishing Order {order} in Time {time.time() - start}")


final_energy_grid_by_order = []
final_energy_grid = np.zeros_like(temp_grid)

for order in range(1, final_order + 1):
    graph_mult = open(f'{nlce_data_dir}/graph_mult_{nlce_type}_{order}.json')
    graph_mult_dict = json.load(graph_mult)
    for graph_id in weight_dict_ordered[order]:
        final_energy_grid += graph_mult_dict[graph_id] * weight_dict[graph_id]

    final_energy_grid_by_order.append(copy.deepcopy(final_energy_grid))


for order in range(6, len(final_energy_grid_by_order)):
    plt.plot(temp_grid, final_energy_grid_by_order[order], label = f"{order + 1}")


plt.legend()
plt.xscale("log")

plt.savefig(f"{proper_property_data_dir}/{property_type}_{final_order}.pdf")
print(f"Total Time: {time.time() - start: .4f}")
