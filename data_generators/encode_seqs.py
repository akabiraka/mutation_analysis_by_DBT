import sys
sys.path.append("../mutation_analysis_by_DBT")

import pandas as pd
from objects.MutationUtils import MutationUtils
from objects.SeqEcodeHelper import SeqEcodeHelper
from Bio import SeqIO

# configurations
pdbs_clean_dir = "data/pdbs_clean/"
fastas_dir = "data/fastas/"
encoded_dir = "data/encoded/"

# inp_file = "data/dataset_5_train.csv"
# out_file = "data/datasets/train.csv"
# inp_file = "data/dataset_5_val.csv"
# out_file = "data/datasets/val.csv"
inp_file = "data/dataset_5_test.csv"
out_file = "data/datasets/test.csv"

cut_sub_seq_len = 15
n_rows_to_skip = 0
n_rows_to_evalutate = 100000

# object initialization
mutation_utils = MutationUtils()
seq_encode_helper = SeqEcodeHelper()

df = pd.read_csv(inp_file)
out_df = pd.DataFrame(columns=["wild", "mutant", "ddg", "label"])

for i, row in df.iterrows():
    if i+1 <= n_rows_to_skip: continue
    
    # extracting the data
    pdb_id, chain_id, mutation, mutation_site, wild_residue, mutant_residue, ddg = mutation_utils.get_row_items(row)
    label = "stabilizing" if ddg>=0 else "destabilizing"

    # creating necessary file paths
    wild_id, mutant_id = pdb_id+chain_id, pdb_id+chain_id+"_"+mutation
    cln_pdb_file = pdbs_clean_dir+wild_id+".pdb"
    wild_fasta_file = fastas_dir+wild_id+".fasta"
    mutant_fasta_file = fastas_dir+mutant_id+".fasta"
    wild_enc_file = encoded_dir+wild_id+".pkl"
    mutant_enc_file = encoded_dir+mutant_id+".pkl"

    # getting 0-based mutation site
    zero_based_mutation_site = mutation_utils.get_zero_based_mutation_site(cln_pdb_file, chain_id, mutation_site)
    print("Row no:{}->{}{}, mutation:{}, 0-based_mutaiton_site:{}".format(i+1, pdb_id, chain_id, mutation, zero_based_mutation_site))

    # getting wild and mutant seq
    wild_seq = next(SeqIO.parse(wild_fasta_file, "fasta")).seq
    mutant_seq = next(SeqIO.parse(mutant_fasta_file, "fasta")).seq

    # getting the sub-sequence around mutation site
    wild_sub_seq = seq_encode_helper.cut_mutation_region(wild_seq, zero_based_mutation_site, cut_sub_seq_len)
    mutant_sub_seq = seq_encode_helper.cut_mutation_region(mutant_seq, zero_based_mutation_site, cut_sub_seq_len)
    print(wild_sub_seq, mutant_sub_seq, mutation, zero_based_mutation_site)
    # print(wild_seq)
    # print(mutant_seq)

    # encoding and saving mutation region by blosum50
    wild_enc = seq_encode_helper.seq_to_blosum(wild_sub_seq)
    mutant_enc = seq_encode_helper.seq_to_blosum(mutant_sub_seq)
    mutation_utils.save_as_pickle(wild_enc, wild_enc_file)
    mutation_utils.save_as_pickle(mutant_enc, mutant_enc_file)
    print(mutation_utils.load_pickle(wild_enc_file).dtype, mutation_utils.load_pickle(mutant_enc_file).dtype)
    print(mutation_utils.load_pickle(wild_enc_file).shape, mutation_utils.load_pickle(mutant_enc_file).shape)
    # print(wild_enc.shape, mutant_enc.shape)

    out_df.loc[i] = [wild_id, mutant_id, ddg, label]

    print()
    if i+1 == n_rows_to_skip+n_rows_to_evalutate: 
        break

out_df.to_csv(out_file, index=False)