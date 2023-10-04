import json, sys, itertools, math
from tqdm import tqdm
import numpy as np
import pynauty

def gen_points_around(node):
    # Takes in node in pseudo-triangular lattice coordinates
    # and returns new nodes around it
    # in tuple coordinate form for everything.

    new_nodes_coordinates = [(node[0] - 1, node[1]), (node[0] - 1, node[1] + 1), (node[0], node[1] + 1)]
    return(new_nodes_coordinates)

def create_bond_info(points):
    # takes in a list of points and returns a list of 
    # bonds and an adjecency dictionary
    adj_dict = {}
    bond_list = []
    for ind, node in enumerate(points):
        adj_dict[ind] = []
        connections = [(node[0] + 1, node[1]), (node[0], node[1] + 1), (node[0] - 1, node[1]), (node[0], node[1] - 1), (node[0] + 1, node[1] + 1), (node[0] - 1, node[1] - 1)]
        for ind_other, node_other in enumerate(points):
            if node_other in connections:
                if ind_other > ind:
                    adj_dict[ind].append(ind_other) 
                    bond_list.append((ind, ind_other, 1))

    return(bond_list, adj_dict)

def iso_hash_function(adj_dict):
    # Takes in the adjecency dict for a graph, then
    # returns its isohash from pynauty

    graph_nauty = pynauty.Graph(len(adj_dict), adjacency_dict = adj_dict)
    certificate = pynauty.certificate(graph_nauty)
    return(hash(certificate))


# Single Site
graph_bond_1 = {"15130871412783076140":()}
graph_mult_1 = {"15130871412783076140":1}
subgraph_mult_1 = {"15130871412783076140":{"15130871412783076140":1}} 

graph_bond_file_1 = open(f'triangle_clusters/graph_bond_triangle_clusters_1.json', 'w')
graph_mult_file_1 = open(f'triangle_clusters/graph_mult_triangle_clusters_1.json', 'w')
subgraph_mult_file_1 = open(f'triangle_clusters/subgraph_mult_triangle_clusters_1.json', 'w')
json.dump(graph_bond_1, graph_bond_file_1)
json.dump(graph_mult_1, graph_mult_file_1)
json.dump(subgraph_mult_1, subgraph_mult_file_1)
graph_bond_file_1.close()
graph_mult_file_1.close()
subgraph_mult_file_1.close()

# Two Site
graph_bond_2 ={"4288071131319860613":[(0,1,1)]}
graph_mult_2 = {"4288071131319860613":3}
subgraph_mult_2 ={"4288071131319860613":{"15130871412783076140":2}}

graph_bond_file_2 = open(f'triangle_clusters/graph_bond_triangle_clusters_2.json', 'w')
graph_mult_file_2 = open(f'triangle_clusters/graph_mult_triangle_clusters_2.json', 'w')
subgraph_mult_file_2 = open(f'triangle_clusters/subgraph_mult_triangle_clusters_2.json', 'w')
json.dump(graph_bond_2, graph_bond_file_2)
json.dump(graph_mult_2, graph_mult_file_2)
json.dump(subgraph_mult_2, subgraph_mult_file_2)
graph_bond_file_2.close()
graph_mult_file_2.close()
subgraph_mult_file_2.close()

iso_cluster = {}
cluster_iso = {}

for order in range(1, 10):
    graph_bond_file = open(f'./Data/NLCE_Data/triangle_symmetric_full/graph_bond_triangle_{order}.json')
    graph_mult_file = open(f'./Data/NLCE_Data/triangle_symmetric_full/graph_mult_triangle_{order}.json')
    subgraph_mult_file = open(f'./Data/NLCE_Data/triangle_symmetric_full/subgraph_mult_triangle_{order}.json')
    
    graph_bond_dict = json.load(graph_bond_file)
    graph_mult_dict = json.load(graph_mult_file)
    subgraph_mult_dict = json.load(subgraph_mult_file)
    
    new_graph_bond = {}
    new_graph_mult = {}
    new_subgraph_mult = {}
    #print(len(graph_bond_dict))

    for cluster in graph_bond_dict:
        #print("-" * 100)
        #print([(node[0], node[1]) for node in graph_bond_dict[cluster][0]])
        points_around = list(set([new_node for node in graph_bond_dict[cluster][0] for new_node in gen_points_around(node)]))
        #print(points_around)
        bond_list, adj_dict = create_bond_info(points_around)
        #print(adj_dict)
        #print("-" * 100)
        iso_hash = iso_hash_function(adj_dict)
        new_graph_bond[iso_hash] = bond_list

        cluster_iso[cluster] = iso_hash
        if iso_hash in new_graph_mult:
            iso_cluster[iso_hash].append(cluster)
            new_graph_mult[iso_hash] += 1
        else:
            iso_cluster[iso_hash] = [cluster]
            temp_dict = {cluster_iso[key]: value[1] for key, value in subgraph_mult_dict[cluster].items()}
            temp_dict["15130871412783076140"] = len(points_around)
            temp_dict["4288071131319860613"] = len(bond_list)
            new_subgraph_mult[iso_hash] = temp_dict
            new_graph_mult[iso_hash] = 1
    
    graph_bond_file = open(f'triangle_clusters/graph_bond_triangle_clusters_{order + 2}.json', 'w')
    graph_mult_file = open(f'triangle_clusters/graph_mult_triangle_clusters_{order + 2}.json', 'w')
    subgraph_mult_file = open(f'triangle_clusters/subgraph_mult_triangle_clusters_{order + 2}.json', 'w')
    json.dump(new_graph_bond, graph_bond_file)
    json.dump(new_graph_mult, graph_mult_file)
    json.dump(new_subgraph_mult, subgraph_mult_file)
    graph_bond_file.close()
    graph_mult_file.close()
    subgraph_mult_file.close()
    #print(iso_cluster)
    #print(new_graph_bond)
    print(f"{order}: {len(new_graph_bond)}")
