import numpy as np
import random
import matplotlib.pyplot as plt
from methods import cra_scores, l3_scores
import networkx as nx
from utils import * 

from methods import l3_scores, cra_scores




# dataset processing
def sampling(graph, test_frac, val_frac, name): 
   
    """
    Splits the edge list into test, validation, and training sets
    based on user-provided fractions.
    Saves each set to a file.
    """
    edge_list = list(graph)
    
    
    random.shuffle(edge_list)  # Shuffle to randomize before splitting

    num_edges = len(edge_list)
    num_test_edges = int(test_frac * num_edges)
    num_val_edges = int(val_frac * num_edges)

    # Split the edges 
    test_set = edge_list[:num_test_edges]
    validation_set = edge_list[num_test_edges:num_test_edges + num_val_edges]
    training_set = edge_list[num_test_edges + num_val_edges:]

    # Write to files
    def write_edges_to_file(edge_list, filepath):
        with open(filepath, 'w') as f:
            for u, v in edge_list:
                f.write(f"{u} {v}\n")

    path = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/dataset")
    path.mkdir(parents = True, exist_ok = True)
    write_edges_to_file(training_set, path / f"train_edges_{name}.txt")
    write_edges_to_file(validation_set, path / f"val_edges_{name}.txt")
    write_edges_to_file(test_set, path / f"test_edges_{name}.txt")

    return training_set, validation_set, test_set




def uniform_sampling(adj_list , test_frac, val_frac):
    """ create two Adjacency Lists from adj_list, train_ael is adj_list without x% of edges,  test_ael contains the x% of edges to guess and a scalar containing the number of edges to guess """
    # create the edge list of the graph -> splits the dataset into training and testing
    edge_list  = []
    for i in adj_list:
        for j in adj_list[i]:
            if j > i:
                edge_list.append((i,j))



    num_test_edges = int(test_frac*len(edge_list)) #test_frac is fraction of edges to hold out for test set 
    test_edge_set = set(random.sample(edge_list, num_test_edges))
    remaining_edges = set(edge_list) - set(test_edge_set) 

    num_val_edges = int(val_frac * len(remaining_edges)) #val_frac is fraction of edges to hold out for validation set
    val_edges = set(random.sample(list(remaining_edges), num_val_edges))
    # print(f"this is the validation edges: {val_edges}")
    # print(f"this is test_edge_set: {test_edge_set}")
    num_edges = len(test_edge_set)
    train_ael = dict()
    test_ael = dict()
    val_ael = dict()

    for i in adj_list:
        # print(f"this is i: {i}")
        for j in adj_list[i]:
            # print(f"this is j: {j}")
            if j > i:
                if (i, j) in val_edges:
                    if i in val_ael and j not in val_ael[i]:
                        val_ael[i].append(j)
                    else:
                        val_ael[i] = [j]
                    if j in val_ael and i not in val_ael[j]:
                        val_ael[j].append(i) 
                    else:
                        val_ael[j] = [i]   
                if (i,j) in test_edge_set:
                    if i in test_ael and j not in test_ael[i]:
                        test_ael[i].append(j)
                    else:
                        test_ael[i] = [j]
                    if j in test_ael and i not in test_ael[j]:
                        test_ael[j].append(i)
                    else:
                        test_ael[j] = [i]
                else:
                    if i in train_ael and j not in train_ael[i]:
                        train_ael[i].append(j)
                    else:
                        train_ael[i] = [j]
                    if j in train_ael and i not in train_ael[j]:
                        train_ael[j].append(i)
                    else:
                        train_ael[j] = [i]   

    return (f"this is train_ael: {train_ael}\n this is test_ael:  {test_ael}\n this is val_ael: {val_ael}")



def all_potential_edges(ael):
    """For each node, adding pairs among its neighbors that are NOT connected directly, to the potential pairs."""

    nodes = list(ael.keys())
    existing_edges = set()
    potential_edges = set() #use a set

    # existing edges
    for u in ael:
        for v in ael[u]:
            edge = tuple(sorted((u, v)))
            existing_edges.add(edge)

    # all possible pairs
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            edge = (u, v)
            if edge not in existing_edges:
                potential_edges.add(edge)

    return potential_edges


def create_potential_adj_list(adj_list):
    """ create an Adjacency List containing for all node k its pairs of unconnected neighbors (= distance-2 pairs) """
    potential_adj_list = {}
    for k, neighbors in adj_list.items():
        potential_links = []
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                node_i, node_j = min(neighbors[i], neighbors[j]) , max(neighbors[i], neighbors[j])    #to have our pair always ordered as (smaller_node, larger_node)
                if node_j not in adj_list[node_i]: # Check that i and j are not connected
                    potential_links.append((node_i, node_j))
        potential_adj_list[k] = potential_links
    return potential_adj_list


def potential_edges_distance3_only(adj_list):
    
    potential_edges = set()
    existing_edges = set()

    # Collect all existing edges (for direct connection check)
    for u in adj_list:
        for v in adj_list[u]:
            existing_edges.add(tuple(sorted((u, v))))

    for u in adj_list:
        for x in adj_list[u]:  # 1 hop from u
            for y in adj_list[x]:  # 2 hops from u
                if y == u:
                    continue
                for v in adj_list[y]:  # 3 hops from u
                    if v != u and v not in adj_list[u]:  # not directly connected
                        edge = tuple(sorted((u, v)))
                        if edge not in existing_edges:
                            potential_edges.add(edge)

    return potential_edges



def second_neighbors(ael):
    sec_neigh = {}
    for node in ael:
        sec_neigh[node] = []
        for neighs in ael[node]:
            for elements in ael[neighs]:
                if elements not in ael[node] and elements != node and elements not in sec_neigh[node]:
                    sec_neigh[node].append(elements)

                continue 
    return f"second neighbors list: {sec_neigh}"






def compute_scaling_factor(learning_file, test_file):
    def count_lines(filepath):
        with open(filepath, 'r') as f:
            return sum(1 for _ in f)

    n_learn = count_lines(learning_file)
    n_test = count_lines(test_file)
    
    if n_learn == 0:
        raise ValueError("Learning set is empty; cannot compute scaling factor.")
    
    s = n_test / n_learn
    print(f"n_learn = {n_learn}")
    print(f"n_test = {n_test}")
    print(f"Scaling factor s = n_test / n_learn = {s:.3f}")
    
    return s



#------------ function to generate the scores of L3 and CRA and create a txt file (training) ---------------
def generate_rank_file(score_dict, true_edges_set, filename):
    """
    Given a dictionary of scores (e.g., from CRA or L3), writes to file:
    node1 node2 label
    where label = 1 if (node1, node2) is in true_edges_set (validation set), else 0
    """
    with open(filename, "w") as f:
        for (u, v), score in sorted(score_dict.items(), key=lambda item: -item[1]):
            label = 1 if (u, v) in true_edges_set or (v, u) in true_edges_set else 0
            f.write(f"{u} {v} {label}\n")


#----------function to generate test ranking files --------------
def generate_cra_testing_file(output_file="cra_testing.txt"):
    from data_processing import load_edges, el_to_ael, potential_edges
    from methods import cra_scores

    # Load training, validation, and test edge lists
    train_edges = load_edges("train_edges.txt")
    val_edges = load_edges("val_edges.txt")
    test_edges = load_edges("test_edges.txt")

    # Build graph from train + val edges
    ael = el_to_ael(train_edges + val_edges)

    # Generate CRA candidates
    candidates = potential_edges(ael)

    # Calculate CRA scores
    scores = cra_scores(candidates, ael)

    # Create lookup set for test edges
    test_edge_set = set((u, v) for u, v in test_edges) | set((v, u) for u, v in test_edges)

    # Write the output file
    with open(output_file, "w") as f:
        for (u, v), score in sorted(scores.items(), key=lambda x: -x[1]):
            label = 1 if (u, v) in test_edge_set else 0
            f.write(f"{u} {v} {label}\n")





def generate_l3_testing_file(output_file="l3_testing.txt"):

    # Load training, validation, and test edge lists
    train_edges = load_edges("train_edges.txt")
    val_edges = load_edges("val_edges.txt")
    test_edges = load_edges("test_edges.txt")

    # Build graph from train + val edges
    ael = el_to_ael(train_edges + val_edges)

    # Calculate L3 scores for all candidate pairs
    scores = l3_scores(ael)

    # Create lookup set for test edges
    test_edge_set = set((u, v) for u, v in test_edges) | set((v, u) for u, v in test_edges)

    # Write the output file
    with open(output_file, "w") as f:
        for (u, v), score in sorted(scores.items(), key=lambda x: -x[1]):
            label = 1 if (u, v) in test_edge_set else 0
            f.write(f"{u} {v} {label}\n")




"""

"we compute a score for every pair of nodes that is not connected in the training graph"

data_path = "/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/data_repository/heinetal-rec.txt"

graph = load_data(data_path)
# print(graph)
print(read_ael(data_path))
# val_edges = load_edges("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankMerging/val_edges.txt")
# train_edges = load_edges("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankMerging/train_edges.txt")
# # ael = el_to_ael(train_edges + val_edges)
# ael = read_ael(graph)

# #--------------- to run CRA and L3 scoring -------------
# # we have already loaded the validation edges and computed scores:
# val_edges = set([tuple(map(int, line.strip().split())) for line in open("val_edges.txt")])

# # Run CRA and L3 scoring
# cra_scores_dict = cra_scores(potential_adj_list(ael), ael)
# l3_scores_dict = l3_scores(ael)

# # Generate learning files
# generate_rank_file(cra_scores_dict, val_edges, "cra_learning.txt")
# generate_rank_file(l3_scores_dict, val_edges, "l3_learning.txt")


# # # Generate learning files
# generate_rank_file(cra_scores_dict, val_edges, "cra_learning.txt")
# generate_rank_file(l3_scores_dict, val_edges, "l3_learning.txt")

"""