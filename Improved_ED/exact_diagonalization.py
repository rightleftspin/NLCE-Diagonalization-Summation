import dask
import sys, time, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.linalg as sl

def get_bit(value, n):
    return ((value >> n & 1) != 0)

def set_bit(value, n):
    return value | (1 << n)

def clear_bit(value, n):
    return value & ~(1 << n)

def load_cluster_bond_info(cluster_bond_info_file_path):
    """
    Read and initialize the graph bond information json file
    """
    # Load the graph bond information into a dictionary
    cluster_bond_info_file = open(cluster_bond_info_file_path)
    cluster_bond_info = json.load(cluster_bond_info_file)

    return(cluster_bond_info)

@dask.delayed
def find_heisenberg_energy_eigenvalues(bond_information, number_sites, tunneling_strength, sparse):
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
    if sparse:
        eigenvalues = spl.eigsh(hamiltonian_matrix, return_eigenvectors=False)
    else:
        eigenvalues = sl.eigvalsh(hamiltonian_matrix.toarray())

    return(eigenvalues)

@dask.delayed
def find_ising_energy_eigenvalues(bond_information, number_sites, tunneling_strength, sparse):
    """
    This function takes the bond information for a specific graph
    and returns its ising eigenvalues
    """
    number_states = 2 ** number_sites
    eigenvalues = []

    for state in range(number_states):
        e_state = 0
        for bond in bond_information:
            if get_bit(state, bond[0]) == get_bit(state, bond[1]):
                e_state -= .25
            else:
                e_state += .25
        eigenvalues.append(e_state)

    return(np.array(eigenvalues))

def write_eigenvalues(eigenvalue_dictionary, order, model, tunneling_strength, base_path):
    """
    This function writes the eigenvalues of a specific model and property
    to the corresponding json file and creates the directory if needed
    """
    # Declare write path
    write_dir = f"{base_path}/{model}"
    write_path = f"{write_dir}/{order}_{''.join([str(j) for j in tunneling_strength])}.json"
    # Create directory if it doesn't exist already
    os.makedirs(write_dir, exist_ok = True)
    # open eigenvalue write file
    eigenvalue_write_file = open(write_path)
    json.dump(eigenvalue_dictionary, eigenvalue_write_file)
    eigenvalue_write_file.close()

    return(write_path)

def ed_main(order, cluster_bond_info, model, tunneling_strength):
    """
    Main function (for external use): Takes in order, model information and
    file path for the graph bond information, then it calculates the
    eigenvalues and returns them to you as an uncomputed dictionary
    """
    sparse = order > 2
    # Load the appropriate graph solver function, preferably
    # utilizing the dask.delayed framework
    solver_function = model_property_functions[model]

    # Load graph eigenvalues in using the solver function
    cluster_eigenvalues = {}
    for cluster_id in cluster_bond_info:
        cluster_eigenvalues[cluster_id] = solver_function(cluster_bond_info[cluster_id], order, tunneling_strength, sparse)

    return(cluster_eigenvalues)

def main(order, cluster_bond_info_file_path, model, tunneling_strength):
    """
    Main function: Takes in order, model information and file path for the graph bond
    information, then it calculates the eigenvalues and returns them to you as an
    uncomputed dictionary
    """
    sparse = order > 2
    # Load the graph bond info dictionary
    cluster_bond_info = load_cluster_bond_info(cluster_bond_info_file_path)
    # Load the appropriate graph solver function, preferably
    # utilizing the dask.delayed framework
    solver_function = model_property_functions[model]

    # Load graph eigenvalues in using the solver function
    cluster_eigenvalues = {}
    for cluster_id in cluster_bond_info:
        cluster_eigenvalues[cluster_id] = solver_function(cluster_bond_info[cluster_id], order, tunneling_strength, sparse)

    return(cluster_eigenvalues)

model_property_functions = {
        "heisenberg_energy": find_heisenberg_energy_eigenvalues,
        "ising_energy": find_ising_energy_eigenvalues
        }

if __name__ == "__main__":
    order = eval(sys.argv[1])
    data_directory = sys.argv[2]
    nlce_type = sys.argv[3]
    cluster_eigenvalues = main(order, f"{data_directory}/{nlce_type}/graph_bond_{nlce_type}_{order}.json", "heisenberg_energy", [1, 0.5])

    start = time.time()
    cluster_eigenvalues_solved = dask.compute(cluster_eigenvalues, scheduler = "processes", num_workers=8, threads_per_worker=1)
    print(f"Elapsed Time: {time.time() - start}")
