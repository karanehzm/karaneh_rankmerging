from methods import *
from data_processing import *
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path
from config import *



def main():

    "===== CONFIG ===== "
    NUM_PREDICION = 100000
    
    "run the rankmerging for the learning phase using the command below"

    # learning_phase(DATASET_NAME, DATA_DIR, TRAIN_DATASET, VAL_DATASET, SCORES_DIR, LEARNING_DIR, PAIRS_DIR)
    "for heinetal:"
    "./Merge_learn 102800318 100000 300 2 l3_learning_heinetal-rec.txt,cra_learning_heinetal-rec.txt ex_100k_heinetal-rec"
    "for human: "
    "./Merge_learn 485664 100000 300 2 l3_learning_human_ppi_lcqb_s900.txt,cra_learning_human_ppi_lcqb_s900.txt ex_100k_human_ppi_lcqb_s900"

    # testing_phase(DATASET_NAME, TRAIN_DATASET, VAL_DATASET, COMBINED_DATASET , TEST_DATASET, SCORES_DIR, LEARNING_DIR, PAIRS_DIR)
    "for heinetal: "
    "./Merge_test 102800318 100000 learning_ex_100k_heinetal-rec.txt 1.5 2 l3_testing_heinetal-rec.txt,cra_testing_heinetal-rec.txt ex_t_100k_heinetal-rec"
    "for human: "
    "./Merge_test 485664 100000 learning_ex_100k_human_ppi_lcqb_s900.txt 1.5 2 l3_testing_human_ppi_lcqb_s900.txt,cra_testing_human_ppi_lcqb_s900.txt ex_t_100k_human_ppi_lcqb_s900"
    
    # "-------------------------------------------- inputs to the rankmerging --------------------------------------"

    CRA_FILE = SCORES_DIR / f"cra_score_test_{DATASET_NAME}.txt"
    L3N_FILE = SCORES_DIR / f"l3_score_test_{DATASET_NAME}.txt"
    rankmerging_output = LEARNING_DIR/f"ranking_ex_t_100k_{DATASET_NAME}.txt"

    # "------------------------------------------ to plot cra, l3n and rankmerging -----------------------------------"
    combined_graph = load_data(COMBINED_DATASET)
    combined_ael = el_to_ael(combined_graph)
    test_set = load_data(TEST_DATASET)
    test_set_ael = el_to_ael(test_set)

    sorted_l3n = read_score_file_as_sorted_list(L3N_FILE)
    sorted_cra = read_score_file_as_sorted_list(CRA_FILE)
    sorted_rank = read_score_file_as_sorted_list(rankmerging_output)
    

    my_num_edges = count_edges(test_set_ael)
   

    my_tp_fp_L3Nf1 = tp_fp(sorted_l3n, test_set_ael, NUM_PREDICION, 500) 
    # print(f"list of true positive: {my_tp_fp_L3Nf1[1]}")
    # print("\n")
    # print(f"list of predicrions: {my_tp_fp_L3Nf1[0]}")
    # print("\n")
    # print(f"list of false positives: {my_tp_fp_L3Nf1[2]}")

    my_pr_rc_L3Nf1 = pr_rc(my_tp_fp_L3Nf1[0], my_tp_fp_L3Nf1[1], my_num_edges)
    to_store_l3n = RESULT_DIR/f"l3n_plot_100k_{DATASET_NAME}"
    # register_image(to_store_l3n, my_pr_rc_L3Nf1[0], my_pr_rc_L3Nf1[1])





    my_tp_fp_cra = tp_fp(sorted_cra, test_set_ael, NUM_PREDICION, 500)
    my_pr_rc_cra = pr_rc(my_tp_fp_cra[0], my_tp_fp_cra[1], my_num_edges)
    to_store_cra = RESULT_DIR/f"cra_plot_100k_{DATASET_NAME}"
    # # # register_image (to_store_cra, my_pr_rc_cra[0], my_pr_rc_cra[1])
   
    
    
    my_tp_fp_rank = tp_fp_rank(rankmerging_output, TEST_DATASET, NUM_PREDICION, 500)
    my_pr_rc_rank = pr_rc(my_tp_fp_rank[0], my_tp_fp_rank[1], my_num_edges)
    to_store_rankmerging = RESULT_DIR/f"rank_plot_100k_{DATASET_NAME}"
    # # # register_image(to_store_rankmerging, my_pr_rc_rank[0], my_pr_rc_rank[1])
    

    curves = [
    ("CRA", my_tp_fp_cra[0], my_tp_fp_cra[1]),
    ("L3N", my_tp_fp_L3Nf1[0], my_tp_fp_L3Nf1[1]),
    ("RankMerging", my_tp_fp_rank[0], my_tp_fp_rank[1])
    ]
    all_plots_path = RESULT_DIR/f"all_plots_100k_{DATASET_NAME}"
    plot_multiple_pr_curves(curves, my_num_edges, all_plots_path)


   



if __name__ == '__main__':
    main()
