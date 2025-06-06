from methods import *
from data_processing import *
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from pathlib import Path




# def main():

    #-------------------------------------- heinetal dataset  -----------------------------------
    # dataset_name = "heinetal"
    # full_data_set = "/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/data_repository/heinetal-rec.txt"
    # full_graph = load_data(full_data_set)
    

    # "split the dataset into three parts ( training, validation, testing)"
    # # sampling(full_graph, 0.2, 0.2, "heinetal") 

                            
    # train_data_set = f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/dataset/train_edges_{dataset_name}.txt"
    # train_graph = load_data(train_data_set)
    # # print(train_graph)

    # " generate candidate pairs from the training graph"
    # train_ael = el_to_ael(train_graph)
    # candidate_pairs_train = potential_edges_distance3_only(train_ael)
    
    
    # " write candidate pairs on a text file (as an edge)"
    # # file_path = "/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/potential_pairs"
    # # write_candidate_pairs_train(candidate_pairs_train, f"{dataset_name}", file_path)
   
    # " write candidate pairs on a file as an adjacency list "
    # # candidate_pairs_train_adj = build_candidate_adj_list(candidate_pairs_train)
    # # write_candidate_adj_train(candidate_pairs_train_adj, f"{dataset_name}", file_path)
    

    # " open the file, read the candidate pairs and calculate the scores and write the scores of the pairs in a file " 
    # " for each pair in the candidate pairs, calculate score then write in a file "
    # " training phase "
    # path = "/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores"
    # candidate_path = f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/potential_pairs/candidate_train_adj_{dataset_name}.txt"
    # pairs_adj = read_candidate_adj_list(candidate_path)
    # cra_score = cra_scores(pairs_adj, train_ael)
    # write_cra_score_to_file(cra_score, path, f"cra_score_{dataset_name}.txt")

   
    # l3N1_score = compute_all_L3Nf1(train_ael)
    # write_l3_score_to_file(l3N1_score, path, f"l3_score_{dataset_name}.txt") #this function writes the calculated scores into a file
    # cra_score_file_name = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores/cra_score_{dataset_name}.txt")
    # l3_score_file_name = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores/l3_score_{dataset_name}.txt")
    # val_edges = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/dataset/val_edges_{dataset_name}.txt")
    # learning_path = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/rankmerging")

    # "creating the learning files to pass as inputs to the rank merging"
    # create_score_learning_txt_file(l3_score_file_name, val_edges, learning_path / f"l3_learning_{dataset_name}.txt")
    # create_score_learning_txt_file(cra_score_file_name, val_edges, learning_path / f"cra_learning_{dataset_name}.txt")
    # train_edges = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/dataset/train_edges_{dataset_name}.txt")
    # combined = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/potential_pairs/combined_{dataset_name}.txt")
"""
    "testing phase"
    "first we need to combine training file and validation file"
    combining_files(train_edges, val_edges, combined)

    test_data_set = f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/potential_pairs/combined_{dataset_name}.txt"
    test_graph = load_data(test_data_set)
    
    "generate candidate pairs for from combined graph"
    test_ael = el_to_ael(test_graph)
    # file_path = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/potential_pairs")

    candidate_pairs_test = potential_edges_distance3_only(test_ael)
    # write_candidate_pairs_test(candidate_pairs_test, f"{dataset_name}", file_path)

    candidate_pairs_test_adj = build_candidate_adj_list(candidate_pairs_test) 
    # write_candidate_adj_test(candidate_pairs_test_adj, f"{dataset_name}", file_path)
    
    "for each pair in the candidate pairs, calculate score then write in a file"
    path = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores")
    cra_score_combined = cra_scores(candidate_pairs_test_adj, test_ael)
    write_cra_score_to_file(cra_score_combined, path, f"cra_score_test_{dataset_name}.txt") 

    l3_score_combined = compute_all_L3Nf1(test_ael)
    write_l3_score_to_file(l3_score_combined, path, f"l3_score_test_{dataset_name}.txt")

    l3_score_test_file_name = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores/l3_score_test_{dataset_name}.txt")
    cra_score_test_file_name = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores/cra_score_test_{dataset_name}.txt")
    test_edges = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/dataset/test_edges_{dataset_name}.txt")
    testing_path = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/rankmerging")
    create_score_testing_txt_file(l3_score_test_file_name, test_edges, testing_path /f"l3_testing_{dataset_name}.txt")
    create_score_testing_txt_file(cra_score_test_file_name, test_edges, testing_path/f"cra_testing_{dataset_name}.txt")
    """
# dataset_name = "heinetal"
"-------------------------------------------- inputs to the rankmerging --------------------------------------"
# cra = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/rankmerging/cra_testing_{dataset_name}.txt")
# l3n = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/rankmerging/l3_testing_{dataset_name}.txt")
# rankmerging = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/rankmerging/ranking_ex_t10M_{dataset_name}.txt")



# files = [cra, l3n]
# N = find_max_node_index(files)
# print(f"the maximum index is: {N}")

# my_plot = plot_precision_recall_curves(rankmerging, cra, l3n)

# result_dir = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/result/plots")
# result_dir.mkdir(parents=True, exist_ok=True)

"full path including filename"
"plotting 4 different sizes of predictions"
# plot_path = result_dir / f"result_10k_pred_{dataset_name}.png"
# plot_path = result_dir / f"result_100k_pred_{dataset_name}.png"
# plot_path = result_dir / f"result_1M_pred_{dataset_name}.png"
#plot_path = result_dir / f"result_10M_pred_{dataset_name}.png"
#plt.savefig(plot_path, dpi=300, bbox_inches='tight')

#plt.close()








def main():
    "----------------------------------------------------- human PPI Dataset --------------------------------------------------------"
    dataset_name = "human"
    full_data_set = "/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/data_repository/human_ppi_lcqb_s900.txt"
    full_graph = load_data(full_data_set)
    

    "split the dataset into three parts ( training, validation, testing)"
    sampling(full_graph, 0.2, 0.2, f"{dataset_name}") 

                            
    train_data_set = f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/dataset/train_edges_{dataset_name}.txt"
    train_graph = load_data(train_data_set)
    # # print(train_graph)

    " generate candidate pairs from the training graph"
    train_ael = el_to_ael(train_graph)
    candidate_pairs_train = potential_edges_distance3_only(train_ael)
    
    
    " write candidate pairs on a text file (as an edge)"
    # file_path = "/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/potential_pairs"
    # write_candidate_pairs_train(candidate_pairs_train, f"{dataset_name}", file_path)
   
    " write candidate pairs on a file as an adjacency list "
    # candidate_pairs_train_adj = build_candidate_adj_list(candidate_pairs_train)
    # write_candidate_adj_train(candidate_pairs_train_adj, f"{dataset_name}", file_path)
    

    " open the file, read the candidate pairs and calculate the scores and write the scores of the pairs in a file " 
    " for each pair in the candidate pairs, calculate score then write in a file "
    " training phase "
    path = "/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores"
    # candidate_path = f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/potential_pairs/candidate_train_adj_{dataset_name}.txt"
    # pairs_adj = read_candidate_adj_list(candidate_path)
    # cra_score = cra_scores(pairs_adj, train_ael)
    # write_cra_score_to_file(cra_score, path, f"cra_score_{dataset_name}.txt")

   
    l3N1_score = compute_all_L3Nf1(train_ael)
    write_l3_score_to_file(l3N1_score, path, f"l3_score_{dataset_name}.txt") #this function writes the calculated scores into a file
    cra_score_file_name = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores/cra_score_{dataset_name}.txt")
    l3_score_file_name = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores/l3_score_{dataset_name}.txt")
    val_edges = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/dataset/val_edges_{dataset_name}.txt")
    learning_path = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/rankmerging")

    # "creating the learning files to pass as inputs to the rank merging"
    # create_score_learning_txt_file(l3_score_file_name, val_edges, learning_path / f"l3_learning_{dataset_name}.txt")
    # create_score_learning_txt_file(cra_score_file_name, val_edges, learning_path / f"cra_learning_{dataset_name}.txt")
    train_edges = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/dataset/train_edges_{dataset_name}.txt")
    combined = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/potential_pairs/combined_{dataset_name}.txt")
    
    "testing phase"
    "first we need to combine training file and validation file"
    combining_files(train_edges, val_edges, combined)

    test_data_set = f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/potential_pairs/combined_{dataset_name}.txt"
    test_graph = load_data(test_data_set)
    
    "generate candidate pairs for from combined graph"
    test_ael = el_to_ael(test_graph)
    file_path = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/potential_pairs")

    # candidate_pairs_test = potential_edges_distance3_only(test_ael)
    # write_candidate_pairs_test(candidate_pairs_test, f"{dataset_name}", file_path)

    # candidate_pairs_test_adj = build_candidate_adj_list(candidate_pairs_test) 
    # write_candidate_adj_test(candidate_pairs_test_adj, f"{dataset_name}", file_path)
    
    # "for each pair in the candidate pairs, calculate score then write in a file"
    # path = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores")
    # cra_score_combined = cra_scores(candidate_pairs_test_adj, test_ael)
    # write_cra_score_to_file(cra_score_combined, path, f"cra_score_test_{dataset_name}.txt") 

    # l3_score_combined = compute_all_L3Nf1(test_ael)
    # write_l3_score_to_file(l3_score_combined, path, f"l3_score_test_{dataset_name}.txt")

    # l3_score_test_file_name = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores/l3_score_test_{dataset_name}.txt")
    # cra_score_test_file_name = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/scores/cra_score_test_{dataset_name}.txt")
    # test_edges = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/dataset/test_edges_{dataset_name}.txt")
    # testing_path = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/rankmerging")
    # create_score_testing_txt_file(l3_score_test_file_name, test_edges, testing_path /f"l3_testing_{dataset_name}.txt")
    # create_score_testing_txt_file(cra_score_test_file_name, test_edges, testing_path/f"cra_testing_{dataset_name}.txt")
    

dataset_name = "human"

"-------------------------------------------- inputs to the rankmerging --------------------------------------"
cra = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/rankmerging/cra_testing_{dataset_name}.txt")
l3n = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/rankmerging/l3_testing_{dataset_name}.txt")
rankmerging = Path(f"/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/rankmerging/ranking_ex_t10M_{dataset_name}.txt")



# files = [cra, l3n]
# N = find_max_node_index(files)
# print(f"the maximum index is: {N}")

my_plot = plot_precision_recall_curves(rankmerging, cra, l3n)

result_dir = Path("/Users/karanehzolfaghari/Desktop/PPI_code/codes_karaneh_ppi/Complementarity_rankmerging/result/plots")
result_dir.mkdir(parents=True, exist_ok=True)

"full path including filename"
"plotting 4 different sizes of predictions"
# plot_path = result_dir / f"result_10k_pred_{dataset_name}.png"
# plot_path = result_dir / f"result_100k_pred_{dataset_name}.png"
# plot_path = result_dir / f"result_1M_pred_{dataset_name}.png"
plot_path = result_dir / f"result_10M_pred_{dataset_name}.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

plt.close()





if __name__ == '__main__':
    main()



