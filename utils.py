from methods import *
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve
import os

# some general functions -------------------------------------------------------
def load_data(data):
    "loading data as Edge list"
    edges = set()
    with open(data , 'r') as file:
        for line in file.readlines():
            nodes = line.strip().split()
            #removes self-loops
            if nodes[0] != nodes[1]:
                first = int(nodes[0])
                second = int(nodes[1])
                edges.add((min(first,second), max(first,second)))

    return edges

       
def read_ael(data):
    "loading data as the adjacency edge list"
    graph_ael = {}
    with open(data, 'r') as file:
        for line in file.readlines():
            node,neighbor, weight =  line.strip().split()
            if node != neighbor:
                if node not in graph_ael:
                    graph_ael[node] = []
                if neighbor not in graph_ael[node]:
                    graph_ael[node].append(neighbor)
                if neighbor not in graph_ael:
                    graph_ael[neighbor] = []
                if node not in graph_ael[neighbor]: 
                    graph_ael[neighbor].append(node)
            
                
    return graph_ael



def count_nodes(graph):
    nodes = []
    for pairs in graph:
        if pairs[0] not in nodes:
            nodes.append(pairs[0])
        if pairs[1] not in nodes:
                nodes.append(pairs[1])
    return len(nodes)

       

def count_edges(graph):
    return len(graph)


def ael_to_el(ael):
    "graph = edge list"
    "ael = adjacency edge list"

    edge_list = set()
    for pairs in ael:
        for element in ael[pairs]:
           print(f"pair is {pairs}, element is {element}")
           pair =tuple((int(pairs), int(element)))
           if tuple(sorted(pair)) not in edge_list:
                edge_list.add(pair)

    return edge_list

   

def el_to_ael(graph):
    """ transform graph to ael """
    ael = dict()
    for node,neigh in graph:
        if node in ael:
            if neigh not in ael[node]:
                ael[node].append(neigh)
        else:
            ael[node] = [neigh]
        if neigh in ael:
            if node not in ael[neigh]:
                ael[neigh].append(node)
        else:
            ael[neigh] = [node]
    return ael


def load_edges(path):
    with open(path) as f:
        edge_list = []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edge_list.append((u, v))
        return edge_list
    

def find_max_node_index(filenames):
    max_index = -1
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    max_index = max(max_index, u, v)
    return max_index + 1  # add 1 to satisfy "strictly larger"



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





def write_candidate_pairs_train(candidate_pairs_train, dataset, file_path):
    """
    Writes candidate pairs of train set (list of (u, v)) to a text file.
    """
    file_path = Path(file_path)
    file_name = f"candidate_train_{dataset}.txt"
    full_path = file_path / file_name
    file_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    with open(full_path, "w") as f:
        for u, v in candidate_pairs_train:
            f.write(f"{u} {v}\n")  # Write candidate pairs as edges


def write_candidate_adj_train(candidate_adj, dataset, file_path):
    "writes candidate pairs of train set as an adjacency list "

    file_path = Path(file_path)
    file_name = f"candidate_train_adj_{dataset}.txt"
    full_path = file_path / file_name
    file_path.mkdir(parents=True, exist_ok = True) #to ensure if the directory exists

    with open(full_path, "w") as f:
        for node in candidate_adj.keys():
            f.write(f"{node} {candidate_adj[node]}\n")
    
     
def write_candidate_pairs_test(candidate_pairs_test, dataset, file_path):
    "Writes candidate pairs of test set (list of (u, v)) to a text file"
   
    file_path = Path(file_path)
    file_name = f"candidate_test_{dataset}.txt"
    full_path = file_path / file_name
    file_path.mkdir(parents=True, exist_ok = True) #to ensure if the directory exists

    with open(full_path, "w") as f:
        for u, v in candidate_pairs_test:
            f.write(f"{u} {v}\n")


def write_candidate_adj_test(candidate_adj, dataset, file_path):
    "writes candidate pairs of test set as an adjacency list "

    file_path = Path(file_path)
    file_name = f"candidate_test_adj_{dataset}.txt"
    full_path = file_path / file_name
    file_path.mkdir(parents=True, exist_ok = True) #to ensure if the directory exists

    with open(full_path, "w") as f:
        for node in candidate_adj.keys():
            f.write(f"{node} {candidate_adj[node]}\n")





def build_candidate_adj_list(candidate_pairs):
    candidate_adj = {}
    for u, v in candidate_pairs:
        if u not in candidate_adj:
            candidate_adj[u] = []
        if v not in candidate_adj:
            candidate_adj[v] = []
        candidate_adj[u].append(v)
        candidate_adj[v].append(u)
    return candidate_adj



# read the candidate adjacency list from a file with the given format.
def read_candidate_adj_list(filepath):
    candidate_adj = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(None, 1)  # Split only on first whitespace
            if len(parts) == 2:
                node = int(parts[0])
                neighbors = eval(parts[1])  # Unsafe for unknown input, but acceptable for trusted data
                candidate_adj[node] = neighbors
    return candidate_adj





def write_cra_score_to_file(cra_scores, path, fileName):
    path = Path(path)
    full_path = path / fileName
    path.mkdir(parents=True, exist_ok = True) #to ensure if the directory exists
    sorted_scores = sorted(cra_scores.items(), key = lambda x : x[1], reverse= True)

    with open(full_path, "w") as f:
        for (u,v), score in sorted_scores:
            line = f"{u} {v} {score}\n"
            f.write(line)



def write_l3_score_to_file(l3_scores, path, fileName):
    path = Path(path)
    full_path = path / fileName
    path.mkdir(parents = True, exist_ok = True) #to ensure if the directory exists
    sorted_scores = sorted(l3_scores.items(), key = lambda x: x[1] , reverse = True)

    with open( full_path, "w") as f:
        for (u,v), score in sorted_scores:
            line = f"{u} {v} {score}\n"
            f.write(line)



def create_score_learning_txt_file(score_file, val_edges_file, learn_file):
    val_edges = set()
    with open(val_edges_file, "r") as f2:
        for line in f2:
            nodes = line.strip().split()
            if len(nodes) >= 2:
                u, v = int(nodes[0]), int(nodes[1])
                edge = tuple(sorted((u,v)))
                val_edges.add(edge)

    with open(score_file, "r") as f1:
        with open(learn_file, "w") as f3:
            for line in f1:
                nodes = line.strip().split()
                if len(nodes) >= 2:
                    u,v = int(nodes[0]) , int(nodes[1])
                    edge = tuple(sorted((u,v)))
                    if edge in val_edges:
                        label = "1"
                        to_write = f"{edge[0]} {edge[1]} {label}\n"
                        f3.write(to_write)
                        

                    else:
                        label = "0"
                        to_write =  f"{edge[0]} {edge[1]} {label}\n"
                        # print(f"edge {edge} does NOT exist and we need to label it as 0")
                        f3.write(to_write)


def combining_files(first_file, second_file, output_file):
    "this function combines two text files (first_file an second_file) then writes them in the output file"
    with open(output_file, "w") as outfile:
        with open(first_file, "r") as f1:
            with open(second_file, "r") as f2:
                for line1 in f1:
                    outfile.write(line1)
            
                for line2 in f2:
                    outfile.write(line2)


def create_score_testing_txt_file(score_file, test_edge_file, test_file):
    val_edges = set()
    with open(test_edge_file, "r") as f2:
        for line in f2:
            nodes = line.strip().split()
            if len(nodes) >= 2:
                u, v = int(nodes[0]), int(nodes[1])
                edge = tuple(sorted((u,v)))
                val_edges.add(edge)

    with open(score_file, "r") as f1:
        with open(test_file, "w") as f3:
            for line in f1:
                nodes = line.strip().split()
                if len(nodes) >= 2:
                    u,v = int(nodes[0]) , int(nodes[1])
                    edge = tuple(sorted((u,v)))
                    if edge in val_edges:
                        label = "1"
                        to_write = f"{edge[0]} {edge[1]} {label}\n"
                        f3.write(to_write)
                        

                    else:
                        label = "0"
                        to_write =  f"{edge[0]} {edge[1]} {label}\n"
                        # print(f"edge {edge} does NOT exist and we need to label it as 0")
                        f3.write(to_write)



def evaluate_from_rankmerging_output(filename, top_k=None):
    total_true_edges = 0
    predicted_true = 0
    TP = 0

    with open(filename, "r") as f:
        lines = f.readlines()

    if top_k:
        lines = lines[:top_k]

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        u, v, score = int(parts[0]), int(parts[1]), int(parts[2])
        if score == 1:
            TP += 1

    # Total number of ground-truth positives in the whole file
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        if int(parts[2]) == 1:
            total_true_edges += 1

    precision = TP / len(lines) if len(lines) > 0 else 0
    recall = TP / total_true_edges if total_true_edges > 0 else 0

    return precision, recall





def evaluate_precision_recall_f1(filename, top_k=100):
    with open(filename, "r") as f:
        lines = [line.strip().split() for line in f if len(line.strip().split()) == 3]

    lines = lines[:top_k]

    total_true_links = sum(1 for line in lines if int(line[2]) == 1)

    TP = 0
    for line in lines:
        if int(line[2]) == 1:
            TP += 1

    precision = TP / top_k if top_k > 0 else 0
    recall = TP / total_true_links if total_true_links > 0 else 0

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score, TP


"function to plot the precision, recall and f1"
def evaluate_precision_recall_f1_curve(filename, max_k=100):
    precisions = []
    recalls = []
    f1_scores = []
    ks = []

    with open(filename, "r") as f:
        lines = [line.strip().split() for line in f if len(line.strip().split()) == 3]

    total_true_links = sum(1 for line in lines if int(line[2]) == 1)

    TP = 0
    for i in range(1, min(len(lines), max_k) + 1):
        u, v, label = lines[i - 1]
        if int(label) == 1:
            TP += 1
        precision = TP / i
        recall = TP / total_true_links if total_true_links > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        ks.append(i)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return ks, precisions, recalls, f1_scores




def read_rank_file(filepath, score_type="rank"):
    y_true = []
    y_scores = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 3:
            try:
                label = int(parts[2])
                if label in [0, 1]:
                    y_true.append(label)
                    if score_type == "rank":
                        # Inverse rank as score: highest score = first line
                        y_scores.append(len(lines) - i)
                    elif score_type == "file_score":
                        # If file already has real-valued scores (e.g., CRA/L3N)
                        y_scores.append(float(parts[2]))
            except ValueError:
                continue

    return y_true, y_scores




def plot_precision_recall_curves(rank_merging_path, cra_path, l3n_path):
    # Read RankMerging output
    y_true_rank, y_scores_rank = read_rank_file(rank_merging_path, score_type="rank")

    # Read CRA scores 
    y_true_cra, y_scores_cra = read_rank_file(cra_path, score_type="rank")

    # Read L3N scores
    y_true_l3n, y_scores_l3n = read_rank_file(l3n_path, score_type="rank")

    # Compute curves
    prec_rank, rec_rank, _ = precision_recall_curve(y_true_rank, y_scores_rank)
    prec_cra, rec_cra, _ = precision_recall_curve(y_true_cra, y_scores_cra)
    prec_l3n, rec_l3n, _ = precision_recall_curve(y_true_l3n, y_scores_l3n)

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(rec_rank, prec_rank, label="RankMerging", linewidth=2)
    plt.plot(rec_cra, prec_cra, label="CRA", linewidth=2)
    plt.plot(rec_l3n, prec_l3n, label="L3N", linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(True)

