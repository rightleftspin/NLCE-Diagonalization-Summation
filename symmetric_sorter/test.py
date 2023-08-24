import json, sys, itertools 
from tqdm import tqdm
import numpy as np

def compare_graphs(graph_bond_vary, graph_bond_keep):
    # Compare two graphs through every permutation to rarrange vary to turn into keep
    truth_array = np.zeros(len(graph_bond_vary))
    for index, bond in enumerate(graph_bond_vary):
        if bond in graph_bond_keep:
            truth_array[index] = 1
        else:
            new_bond = [bond[1], bond[0], bond[2]]
            if new_bond in graph_bond_keep:
                truth_array[index] = 1

    if np.sum(truth_array) == len(graph_bond_vary):
        return(True)
    else:
        return(False)
    

number = int(sys.argv[2])
nlce_type = sys.argv[1]
graph_file = open(f'{nlce_type}/graph_bond_{nlce_type}_{number}.json')
graph_dict = json.load(graph_file)
            
for graph_id, bond_info in tqdm(graph_dict.items()):
    for permutation in itertools.permutations(range(number)):
        bond_trial = []
        for bond in bond_info[1]:
            bond_trial.append([permutation[bond[0]], permutation[bond[1]], bond[2]])
        
        if compare_graphs(bond_trial, bond_info[2]):
            #print("-" * 10)
            #print("Passed")
            #print(permutation)
            #print(bond_info[1])
            #print(bond_trial)
            #print(bond_info[2])
            #print(f"{graph_id}: again {permutation}")
            #print("-" * 10)
            temp_bond_info = [0] * len(bond_info[0])
            for ind, new in enumerate(permutation):
                temp_bond_info[new] = bond_info[0][ind]

            graph_dict[graph_id] = (temp_bond_info, bond_info[-1])
            break

write_file = open(f'{nlce_type}_alt/graph_bond_{nlce_type}_{number}.json', 'w')
json.dump(graph_dict, write_file)
write_file.close()


