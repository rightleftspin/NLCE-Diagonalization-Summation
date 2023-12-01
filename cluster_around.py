import json, sys, itertools, math
from tqdm import tqdm
import numpy as np
import pynauty

def find_subgraphs(iso_dict, subgraph_info, cluster, subgraph):
    # Takes in a cluster and one of its subclusters to add it to its subgraph dictionary
    # fills out subgraph
    # Initialize an empty graph dictionary
    clusterGraph = {}
    subgraph = list(subgraph)
    for index, vertex in enumerate(subgraph):
        # Initialize the vertex with an empty set
        clusterGraph[index] = set()
        # Loop over all the possible adjecent verticies
        for adjVertex in cluster[vertex]:
            # Check if the vertex is in the cluster
            if adjVertex in subgraph:
                # add to the graph as an edge if it is
                clusterGraph[index].add(subgraph.index(adjVertex))
    #hashes and adds up subgraph
    subgraph_hash = iso_hash_function(clusterGraph)
    if subgraph_hash in iso_dict:
        if subgraph_hash in subgraph_info:
            subgraph_info[subgraph_hash] += 1
        else:
            subgraph_info[subgraph_hash] = 1

    return()

def vSimple(graph, subgraph, neighbors, guardingSet, size, graphFunc):
    # This is a recursive algorithm that takes a graph and breaks it up into all possible subgraphs
    # of a specific order labelled by the size

    # Start by checking to see if your subgraph is already the proper size
    if len(subgraph) == size:
        # If it is, try adding it to the graph dictionary
      #  print(size, subgraph)
        graphFunc(subgraph)    
        return(True)

    hasIntLeaf = False
    # Loop over all the neighbors for the current node
    neighborIterator = neighbors.copy()
    guardingSetClone = guardingSet.copy()

    while (len(neighborIterator) > 0):
        # create a subgraph by adding the neighbor
        neighbor = neighborIterator.pop()
        newSubgraph = subgraph | {neighbor}
        # Add all the neighbors of the new node that we just added to the new subgraph and
        # take out any neighbors in the guarding set and that are already in the subgraph
        addNeighbors = graph[neighbor].difference(subgraph).difference(guardingSetClone)
        # Create a new set of neighbors that is a combination of the old set of neighbors
        # and the new set of neighbors from the new node
        newNeighbors = neighborIterator | addNeighbors
        # Recursively call this algorithm again with the new subgraph and new neighbor set
        end = vSimple(graph, newSubgraph, newNeighbors, guardingSetClone, size, graphFunc)
        if end:
            hasIntLeaf = True
        else:
            return(hasIntLeaf)
        # Update the guarding set to include the neighbor
        guardingSetClone.add(neighbor)

        # If the guarding set ever gets too big, break out of this loop
        if (len(graph) - len(guardingSetClone)) < size:
            return(hasIntLeaf)

    return(hasIntLeaf)

def enumerateGraph(graph, size, startingVertices, graphFunc):
    # This is a simple wrapper function for the recursive subgraph generator, vSimple
    # Initialize with an empty guarding set
    guardingSet = set()

    for vertex in startingVertices:
        # Initialize with the neighbors of the starting vertex
        neighbors = graph[vertex]
        # start running vSimple
        vSimple(graph, {vertex}, neighbors.difference(guardingSet), guardingSet.copy(), size, graphFunc)
        #print(guardingSet)
        # add vertex to guardingSet
        guardingSet = guardingSet | {vertex}

    return(None)

def gen_points_around(node):
    # Takes in node in pseudo-triangular lattice coordinates
    # and returns new nodes around it
    # in tuple coordinate form for everything.

    new_nodes_coordinates = [(node[0], node[1]), (node[0] - 1, node[1]), (node[0], node[1] + 1)]
    return(new_nodes_coordinates)

def create_bond_info(points):
    # takes in a list of points and returns a list of 
    # bonds and an adjecency dictionary
    adj_dict = {}
    bond_list = []
    for ind, node in enumerate(points):
        adj_dict[ind] = set()
        connections = [(node[0] + 1, node[1]), (node[0], node[1] + 1), (node[0] - 1, node[1]), (node[0], node[1] - 1), (node[0] - 1, node[1] - 1), (node[0] + 1, node[1] + 1)]
        for ind_other, node_other in enumerate(points):
            if node_other in connections:
                adj_dict[ind].add(ind_other) 
                if ind_other > ind:
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

graph_bond_file_1 = open(f'triangle_clusters/graph_bond_triangle_1.json', 'w')
graph_mult_file_1 = open(f'triangle_clusters/graph_mult_triangle_1.json', 'w')
subgraph_mult_file_1 = open(f'triangle_clusters/subgraph_mult_triangle_1.json', 'w')
json.dump(graph_bond_1, graph_bond_file_1)
json.dump(graph_mult_1, graph_mult_file_1)
json.dump(subgraph_mult_1, subgraph_mult_file_1)
graph_bond_file_1.close()
graph_mult_file_1.close()
subgraph_mult_file_1.close()

# Two Site
#graph_bond_2 ={"4288071131319860613":[(0,1,1)]}
#graph_mult_2 = {"4288071131319860613":3}
#subgraph_mult_2 ={"4288071131319860613":{"15130871412783076140":2}}
#
#graph_bond_file_2 = open(f'triangle_clusters/graph_bond_triangle_clusters_2.json', 'w')
#graph_mult_file_2 = open(f'triangle_clusters/graph_mult_triangle_clusters_2.json', 'w')
#subgraph_mult_file_2 = open(f'triangle_clusters/subgraph_mult_triangle_clusters_2.json', 'w')
#json.dump(graph_bond_2, graph_bond_file_2)
#json.dump(graph_mult_2, graph_mult_file_2)
#json.dump(subgraph_mult_2, subgraph_mult_file_2)
#graph_bond_file_2.close()
#graph_mult_file_2.close()
#subgraph_mult_file_2.close()

iso_cluster = {}
cluster_iso = {}

for order in range(1, 8):
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
    #print(graph_bond_dict)

    for cluster in graph_bond_dict:
        #print("-" * 100)
        #print([(node[0], node[1]) for node in graph_bond_dict[cluster][0]])
        points_around = list(set([new_node for node in graph_bond_dict[cluster][0] for new_node in gen_points_around(node)]))
        #print(points_around)
        bond_list, adj_dict = create_bond_info(points_around)
        #print(adj_dict)
        #print("-" * 100)
        iso_hash = iso_hash_function(adj_dict)

        cluster_iso[cluster] = iso_hash
        if iso_hash in new_graph_mult:
            iso_cluster[iso_hash].append(cluster)
            new_graph_mult[iso_hash] += 1
        else:
            new_graph_bond[iso_hash] = (bond_list, points_around)
            iso_cluster[iso_hash] = [cluster]
            temp_dict = {}
            graph_func_temp = lambda subgraph: find_subgraphs(iso_cluster, temp_dict, adj_dict, subgraph)
            #print(f"adj dict {adj_dict}")
            for i in range(3, len(adj_dict)):
     #           print(adj_dict)
                #print("test")
                #print(list(range(len(adj_dict))))
                enumerateGraph(adj_dict, i, list(range(len(adj_dict))), graph_func_temp)
            #print(f"temp dict {temp_dict}")
            #temp_dict = {cluster_iso[key]: value[1] for key, value in subgraph_mult_dict[cluster].items()}
            print(new_graph_mult.keys())
            print(temp_dict.keys())
            for subgraph in temp_dict:
                if subgraph in new_graph_mult:
                    print(subgraph)
                    print(new_graph_bond[subgraph][1])
                    print(points_around)
            temp_dict["15130871412783076140"] = len(points_around)
            #temp_dict["4288071131319860613"] = len(bond_list)
            new_subgraph_mult[iso_hash] = temp_dict
            new_graph_mult[iso_hash] = 1
    
    graph_bond_file = open(f'triangle_clusters/graph_bond_triangle_{order + 1}.json', 'w')
    graph_mult_file = open(f'triangle_clusters/graph_mult_triangle_{order + 1}.json', 'w')
    subgraph_mult_file = open(f'triangle_clusters/subgraph_mult_triangle_{order + 1}.json', 'w')
    json.dump(new_graph_bond, graph_bond_file)
    json.dump(new_graph_mult, graph_mult_file)
    json.dump(new_subgraph_mult, subgraph_mult_file)
    graph_bond_file.close()
    graph_mult_file.close()
    subgraph_mult_file.close()
    #print(iso_cluster)
    #print(new_graph_bond)
    print(f"{order}: {len(new_graph_bond)}")
