import dask
import sys, time, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

def get_bit(value, n):
    return ((value >> n & 1) != 0)

def set_bit(value, n):
    return value | (1 << n)

def clear_bit(value, n):
    return value & ~(1 << n)

def load_graph_bond_info(graph_bond_info_file_path):
    """
    Read and initialize the graph bond information json file
    """
    # Load the graph bond information into a dictionary
    graph_bond_info_file = open(graph_bond_info_file_path) 
    graph_bond_info = json.load(graph_bond_info_file)

    return(graph_bond_info)

@dask.delayed
def find_heisenberg_energy_eigenvalues(bond_information, number_sites, tunneling_strength):
    """
    This function takes the bond information for a specific graph
    and returns its heisenberg hamiltonian matrix in sparse DOK form.
    """
    # Initialize variables and empty hamiltonian matrix
    number_states = 2 ** number_sites
    rows, columns, data = [], [], []

    # loop over all 2 ^ N - 1 states
    for state in range(number_states):
        # loop over all bonds between particles in the state
        for bond in bond_information:
            # if the sites are the same
            if get_bit(state, bond[0]) == get_bit(state, bond[1]):
                # Then add only the diagonal term j/4 (divide by 4 for the spin of electron)
                rows.append(state)
                columns.append(state)
                data.append(tunneling_strength[bond[2] - 1] / 4)
            else:
                # Then subtract the diagonal term j/4 and also add an off diagonal term
                rows.append(state)
                columns.append(state)
                data.append(-(tunneling_strength[bond[2] - 1] / 4))
                # Flip the bits of the number that corresponds to the sites
                if get_bit(state, bond[0]):
                    changed_state = set_bit(clear_bit(state, bond[0]), bond[1])
                else:
                    changed_state = clear_bit(set_bit(state, bond[0]), bond[1])

                rows.append(state)
                columns.append(changed_state)
                data.append(.5 * tunneling_strength[bond[2] - 1])

    # Construct hamiltonian matrix of csr sparse type
    hamiltonian_matrix = sp.csr_matrix((data, (rows, columns)), (number_states, number_states))

    # Solve for the eigenvalues of the hamiltonian matrix
    eigenvalues = spl.eigsh(hamiltonian_matrix, return_eigenvectors=False)

    return(eigenvalues)

def main(order, graph_bond_info_file_path, model, tunneling_strength):
    """
    Main function: Takes in order, model information and file path for the graph bond
    information, then it calculates the eigenvalues and returns them to you as an
    uncomputed dictionary
    """
    # Load the graph bond info dictionary
    graph_bond_info = load_graph_bond_info(graph_bond_info_file_path)
    # Load the appropriate graph solver function, preferably
    # utilizing the dask.delayed framework
    solver_function = model_property_functions[model]

    # Load graph eigenvalues in using the solver function
    graph_eigenvalues = {}
    for graph in graph_bond_info:
        graph_eigenvalues[graph] = solver_function(graph_bond_info[graph], order, tunneling_strength)

    return(graph_eigenvalues)

model_property_functions = {
        "heisenberg_energy": find_heisenberg_energy_eigenvalues
        }

if __name__ == "__main__":
    order = eval(sys.argv[1])
    data_directory = sys.argv[2]
    nlce_type = sys.argv[3]
    graph_eigenvalues = main(order, f"{data_directory}/{nlce_type}/graph_bond_{nlce_type}_{order}.json", "heisenberg_energy", [1])

    start = time.time()
    graph_eigenvalues_solved = dask.compute(graph_eigenvalues, scheduler = "processes", num_workers=8, threads_per_worker=1)
    print(f"Elapsed Time: {time.time() - start}")
