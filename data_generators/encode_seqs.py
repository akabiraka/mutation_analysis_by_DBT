import sys
sys.path.append("../mutation_analysis_by_DBT")

import pandas as pd
from objects.MutationUtils import MutationUtils
from objects.SeqEcodeHelper import SeqEcodeHelper
from Bio import SeqIO

# configurations
pdbs_clean_dir = "data/pdbs_clean/"
fastas_dir = "data/fastas/"
cut_sub_seq_len = 15

input_file_path = "data/dataset_5_train.csv"
# input_file_path = "data/dataset_5_test.csv"
n_rows_to_skip = 0
n_rows_to_evalutate = 10#0000

# object initialization
mutation_utils = MutationUtils()
seq_encode_helper = SeqEcodeHelper()

df = pd.read_csv(input_file_path)

for i, row in df.iterrows():
    if i+1 <= n_rows_to_skip: continue
    
    # extracting the data
    pdb_id, chain_id, mutation, mutation_site, wild_residue, mutant_residue, ddg = mutation_utils.get_row_items(row)

    # creating necessary file paths
    cln_pdb_file = pdbs_clean_dir+pdb_id+chain_id+".pdb"
    wild_fasta_file = fastas_dir+pdb_id+chain_id+".fasta"
    mutant_fasta_file = fastas_dir+pdb_id+chain_id+"_"+mutation+".fasta"

    # getting 0-based mutation site
    zero_based_mutation_site = mutation_utils.get_zero_based_mutation_site(cln_pdb_file, chain_id, mutation_site)

    # getting wild and mutant seq
    wild_seq = next(SeqIO.parse(wild_fasta_file, "fasta")).seq
    mutant_seq = next(SeqIO.parse(mutant_fasta_file, "fasta")).seq

    # getting the sub-sequence around mutation site
    wild_sub_seq = seq_encode_helper.cut_mutation_region(wild_seq, zero_based_mutation_site, cut_sub_seq_len)
    mutant_sub_seq = seq_encode_helper.cut_mutation_region(mutant_seq, zero_based_mutation_site, cut_sub_seq_len)
    print(wild_sub_seq, mutant_sub_seq, mutation, zero_based_mutation_site)
    # print(wild_seq)
    # print(mutant_seq)

    # encoding mutation region by blosum50
    wild_enc = seq_encode_helper.seq_to_blosum(wild_sub_seq)
    mutant_enc = seq_encode_helper.seq_to_blosum(mutant_sub_seq)
    print(wild_enc.shape, mutant_enc.shape)
    
    print()
    if i+1 == n_rows_to_skip+n_rows_to_evalutate: 
        break