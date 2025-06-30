from methods import *
from pathlib import Path
from config import *
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
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

   

# def el_to_ael(graph):
#     """ transform graph to ael """
#     ael = dict()
#     for node,neigh in graph:
#         if node in ael:
#             if neigh not in ael[node]:
#                 ael[node].append(neigh)
#         else:
#             ael[node] = [neigh]
#         if neigh in ael:
#             if node not in ael[neigh]:
#                 ael[neigh].append(node)
#         else:
#             ael[neigh] = [node]
#     return ael

def el_to_ael(graph):
    """Transform edge list to adjacency list (ael) with string node IDs and undirected structure"""
    ael = dict()
    for u, v in graph:
        u, v = str(u), str(v)  # 🔑 ensure consistent format
        if u not in ael:
            ael[u] = set()
        if v not in ael:
            ael[v] = set()
        ael[u].add(v)
        ael[v].add(u)
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





def write_candidate_pairs_train(candidate_pairs_train, dataset, file_path, force = False):
    """
    Writes candidate pairs of train set (list of (u, v)) to a text file.
    """
    file_path = Path(file_path)
    file_name = f"candidate_train_{dataset}.txt"
    full_path = file_path / file_name
    file_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    if full_path.exists() and not force: 
        print(f"File {full_path} already exists. Skipping write.")
    
    with open(full_path, "w") as f:
        for u, v in candidate_pairs_train:
            f.write(f"{u} {v}\n")  # Write candidate pairs as edges


def write_candidate_adj_train(candidate_adj, dataset, file_path, force = False):
    "writes candidate pairs of train set as an adjacency list "

    file_path = Path(file_path)
    file_name = f"candidate_train_adj_{dataset}.txt"
    full_path = file_path / file_name
    file_path.mkdir(parents=True, exist_ok = True) #to ensure if the directory exists

    if full_path.exists() and not force: 
        print(f"File {full_path} already exists. Skipping write.")

    with open(full_path, "w") as f:
        for node in candidate_adj.keys():
            f.write(f"{node} {candidate_adj[node]}\n")
    
     
def write_candidate_pairs_test(candidate_pairs_test, dataset, file_path, force = False):
    "Writes candidate pairs of test set (list of (u, v)) to a text file"
   
    file_path = Path(file_path)
    file_name = f"candidate_test_{dataset}.txt"
    full_path = file_path / file_name
    file_path.mkdir(parents=True, exist_ok = True) #to ensure if the directory exists

    if full_path.exists() and not force: 
        print(f"File {full_path} already exists. Skipping write.")
       
    
    with open(full_path, "w") as f:
        for u, v in candidate_pairs_test:
            f.write(f"{u} {v}\n")


def write_candidate_adj_test(candidate_adj, dataset, file_path, force = False):
    "writes candidate pairs of test set as an adjacency list "

    file_path = Path(file_path)
    file_name = f"candidate_test_adj_{dataset}.txt"
    full_path = file_path / file_name
    file_path.mkdir(parents=True, exist_ok = True) #to ensure if the directory exists

    if full_path.exists() and not force: 
        print(f"File {full_path} already exists. Skipping write.")
       
    
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





def write_cra_score_to_file(cra_scores, path, fileName, force = False):
    path = Path(path)
    full_path = path / fileName
    path.mkdir(parents=True, exist_ok = True) #to ensure if the directory exists
    if full_path.exists() and not force:
        print(f"File {full_path} already exists. Skipping write.")
       
    
    sorted_scores = sorted(cra_scores.items(), key = lambda x : x[1], reverse= True)
    with open(full_path, "w") as f:
        for (u,v), score in sorted_scores:
            line = f"{u} {v} {score}\n"
            f.write(line)



def write_l3_score_to_file(l3_scores, path, fileName, force = False):
    path = Path(path)
    full_path = path / fileName
    path.mkdir(parents = True, exist_ok = True) #to ensure if the directory exists
    if full_path.exists() and not force:
        print(f"File {full_path} already exists. Skipping write.")
       
    
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





def read_score_file_as_sorted_list(path):
    """Reads (u, v, score) lines and returns sorted list of ((u,v), score) tuples descending by score"""
    score_list = []
    with open(path, "r") as f:
        for line in f:
            u, v, score = line.strip().split()
            # score_list.append(((int(u), int(v)), float(score)))
            score_list.append((((u),(v)), float(score)))
    return sorted(score_list, key=lambda x: x[1], reverse=True)





def plot_all_pr_curves(y_trues, y_scores, labels, filename):
    plt.figure(figsize=(10, 8))
    for y_true, y_score, label in zip(y_trues, y_scores, labels):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Comparison')
    plt.legend()
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(filename)
    plt.show()


def sorting_scores(score_tab):
    """ generate a list of pairs by decreasing score value from a dictionary of scores of the form d[(i,j)]=score """
    # pre_sorting =  sorted(score_tab.items(), key=lambda v: random.random())
    items = list(score_tab.items())
    random.shuffle(items)
    # sorted_scores = sorted(pre_sorting, key=lambda x: x[1], reverse=True)  
    return sorted(items, key=lambda x: x[1], reverse=True)



def tp_fp(score_lst, test_ael, num_pred_max, step):
    """ generate a triplet of lists with the number of predictions, number of TP predictions , number of FP predictions every step until num_pred_max """
    list_tp = []
    list_fp = []
    list_pred = []
   
    num_pred = 0
    num_tp = 0
    num_fp = 0
    
    for pred in score_lst[:num_pred_max+1] :
        (i,j), score = pred
        if i in test_ael and j in test_ael[i]:
            num_tp += 1
        else:
            num_fp += 1
        num_pred +=1
    
        if (num_pred-1) % step == 0:
            list_tp.append(num_tp)
            list_fp.append(num_fp)
            list_pred.append(num_pred)
    return list_pred, list_tp, list_fp
    



def pr_rc(list_pred, list_tp, num_edges):
    """ generate a couple of lists with the precision and recall for every number of prediction"""
    list_pr = []
    list_rc = []
    for i in range(len(list_pred)):
        list_pr.append(list_tp[i]/list_pred[i])
        list_rc.append(list_tp[i]/num_edges)
    return list_rc, list_pr  


def tp_fp_rank(rank_file_path, test_set_path, num_pred_max, step):
    """
    Generate cumulative prediction stats (num predictions, TP, FP)
    every `step` from a RankMerging output file.
    """

    # Load ground truth test set into a dictionary-like structure
    test_set = set()
    with open(test_set_path, "r") as f:
        for line in f:
            i, j = map(int, line.strip().split())
            test_set.add((i, j))
            test_set.add((j, i))  # assume undirected graph

    list_tp = []
    list_fp = []
    list_pred = []

    num_tp = 0
    num_fp = 0
    num_pred = 0

    with open(rank_file_path, "r") as f:
        for line in f:
            if num_pred >= num_pred_max:
                break

            i, j, label = line.strip().split()
            pair = (int(i), int(j))
            label = int(label)

            if label == 1 and pair in test_set:
                num_tp += 1
            else:
                num_fp += 1

            num_pred += 1

            if (num_pred - 1) % step == 0:
                list_pred.append(num_pred)
                list_tp.append(num_tp)
                list_fp.append(num_fp)

    return list_pred, list_tp, list_fp




def show_pr (tab1, tab2):
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.xticks(np.arange(0, 1, 0.05))
    plt.yticks(np.arange(0, 1, 0.25))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot (tab1 , tab2 , marker = '.' , linewidth=1)
    plt.show()

def register_image (filename, tab1, tab2):
    plt.figure(figsize=(12,12))
    plt.plot (tab1 , tab2 , marker = '.' , linewidth=1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 1)  
    plt.xticks(np.arange(0, 1.01, 0.05))
    plt.yticks(np.arange(0, 1.01, 0.25))
    plt.savefig(filename, bbox_inches = "tight")
    plt.show()





def plot_multiple_pr_curves(curves, num_edges, save_path):

    plt.figure(figsize=(8, 6))

    for label, list_pred, list_tp in curves:
        precision = [tp / pred if pred != 0 else 0 for tp, pred in zip(list_tp, list_pred)]
        recall = [tp / num_edges for tp in list_tp]
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{label} (AUC = {pr_auc:.3f})", linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Combined Precision-Recall Curve")
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()




def pr_rec_rank(rank_file_path, test_set_path, step):
    # Load the ground truth test set into a set of pairs
    test_set = set()
    with open(test_set_path, "r") as f:
        for line in f:
            i, j = line.strip().split()
            test_set.add((int(i), int(j)))
            test_set.add((int(j), int(i)))  # optional: make undirected

    num_tp = 0
    num_fp = 0
    num_pred = 0

    list_pred = []  # number of predictions
    list_tp = []    # number of true positives
    list_fp = []    # number of false positives

    with open(rank_file_path, "r") as f:
        for line in f:
            i, j, label = line.strip().split()
            pair = (int(i), int(j))
            label = int(label)
            num_pred += 1

            if label == 1 and pair in test_set:
                num_tp += 1
            else:
                num_fp += 1

            if num_pred % step == 0:
                list_pred.append(num_pred)
                list_tp.append(num_tp)
                list_fp.append(num_fp)

    return list_pred, list_tp, list_fp




def learning_phase(dataset_name, data_dir, train_dataset, val_dataset, scores_dir, learning_dir, pairs_dir):
    from data_processing import potential_edges_distance3_only, sampling


    full_data_set = data_dir / f"{dataset_name}.txt"
    full_graph = load_data(full_data_set)
    
    
    "split the dataset into three parts ( training, validation, testing)"
    sampling(full_graph, 0.2, 0.2, dataset_name) 

                            
    "load the data as pairs"
    train_graph = load_data(train_dataset)
    

    " generate candidate pairs from the training graph"
    train_ael = el_to_ael(train_graph)
    candidate_pairs_train = potential_edges_distance3_only(train_ael)
    
    
    " write candidate pairs on a text file (as an edge)"
    write_candidate_pairs_train(candidate_pairs_train, f"{dataset_name}", pairs_dir, force = False)
   
    " write candidate pairs on a file as an adjacency list "
    candidate_pairs_train_adj = build_candidate_adj_list(candidate_pairs_train)
    write_candidate_adj_train(candidate_pairs_train_adj, f"{dataset_name}", pairs_dir, force = False)
    

    " open the file, read the candidate pairs and calculate the scores and write the scores of the pairs in a file " 
    " for each pair in the candidate pairs, calculate score then write in a file "

    " ----------------------------------------------------- training phase ----------------------------------------------------------"
    from methods import cra_scores, compute_all_L3Nf1
    candidate_path = pairs_dir/f"candidate_train_adj_{dataset_name}.txt"
    pairs_adj = read_candidate_adj_list(candidate_path)
    cra_score = cra_scores(pairs_adj, train_ael)
    write_cra_score_to_file(cra_score, scores_dir, f"cra_score_{dataset_name}.txt", force = False)

   
    l3N1_score = compute_all_L3Nf1(train_ael)
    write_l3_score_to_file(l3N1_score, scores_dir, f"l3_score_{dataset_name}.txt", force = False) #this function writes the calculated scores into a file

    cra_file = scores_dir / f"cra_score_{dataset_name}.txt"
    l3n_file = scores_dir / f"l3_score_{dataset_name}.txt"

    "creating the learning files to pass as inputs to the rank merging"
    create_score_learning_txt_file(l3n_file, val_dataset, learning_dir / f"l3_learning_{dataset_name}.txt")
    create_score_learning_txt_file(cra_file, val_dataset, learning_dir / f"cra_learning_{dataset_name}.txt")
   

    "calculate the maximum node pair"
    files = [cra_file, l3n_file]
    print(f"maximum number of nodes: {find_max_node_index(files)}") # 102800318
    print(f"learning files: l3_learning_{dataset_name}.txt , cra_learning_{dataset_name}.txt")




def testing_phase(dataset_name, train_dataset, val_dataset, combined_dataset , test_dataset, scores_dir, learning_dir, pairs_dir):

    "first we need to combine training file and validation file, find candidate pairs from them, then see if they exist on test set"

    combining_files(train_dataset, val_dataset, combined_dataset)
   
   
    combined_graph = load_data(combined_dataset)
    combined_ael = el_to_ael(combined_graph)
    test_set = load_data(test_dataset)
    test_set_ael = el_to_ael(test_set)

    "generate candidate pairs from the combined graph"
    from data_processing import potential_edges_distance3_only
    candidate_pairs_test = potential_edges_distance3_only(combined_ael)
    write_candidate_pairs_test(candidate_pairs_test, f"{dataset_name}", pairs_dir, force = False)

    candidate_pairs_test_adj = build_candidate_adj_list(candidate_pairs_test) 
    write_candidate_adj_test(candidate_pairs_test_adj, f"{dataset_name}", pairs_dir, force = False)
    
    "for each pair in the candidate pairs, calculate score then write in a file"
    from methods import cra_scores, compute_all_L3Nf1
    cra_score_combined = cra_scores(candidate_pairs_test_adj, combined_ael)
    write_cra_score_to_file(cra_score_combined, scores_dir, f"cra_score_test_{dataset_name}.txt", force = False) 

    l3_score_combined = compute_all_L3Nf1(combined_ael)
    write_l3_score_to_file(l3_score_combined, scores_dir, f"l3_score_test_{dataset_name}.txt", force = False)
 



    l3_score_test_file_name = scores_dir /f"l3_score_test_{dataset_name}.txt"
    cra_score_test_file_name = scores_dir / f"cra_score_test_{dataset_name}.txt"

    create_score_testing_txt_file(l3_score_test_file_name, test_dataset, learning_dir /f"l3_testing_{dataset_name}.txt")
    create_score_testing_txt_file(cra_score_test_file_name, test_dataset, learning_dir /f"cra_testing_{dataset_name}.txt")

