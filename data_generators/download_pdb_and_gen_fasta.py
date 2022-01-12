import sys
sys.path.append("../mutation_analysis_by_DBT")

import csv
import pandas as pd
import numpy as np
import torch
from objects.PDBData import PDBData
from objects.Selector import ChainAndAminoAcidSelect
from objects.MutationUtils import MutationUtils

# configurations
pdb_dir = "data/pdbs/"
pdbs_clean_dir = "data/pdbs_clean/"
fastas_dir = "data/fastas/"
CIF = "mmCif"

# input_file_path = "data/dataset_5_train.csv"
input_file_path = "data/dataset_5_test.csv"
n_rows_to_skip = 0
n_rows_to_evalutate = 100000

# object initialization
PDBData = PDBData(pdb_dir=pdb_dir)
mutation_utils = MutationUtils()

df = pd.read_csv(input_file_path)

for i, row in df.iterrows():
    if i+1 <= n_rows_to_skip: continue
    
    # extracting the data
    pdb_id, chain_id, mutation, mutation_site, wild_residue, mutant_residue, ddg = mutation_utils.get_row_items(row)

    # creating necessary file paths
    cln_pdb_file = pdbs_clean_dir+pdb_id+chain_id+".pdb"
    wild_fasta_file = fastas_dir+pdb_id+chain_id+".fasta"
    mutant_fasta_file = fastas_dir+pdb_id+chain_id+"_"+mutation+".fasta"

    # downloading and cleaning PDB structure
    PDBData.download_structure(pdb_id=pdb_id)
    PDBData.clean(pdb_id=pdb_id, chain_id=chain_id, selector=ChainAndAminoAcidSelect(chain_id))

    # getting 0-based mutation site
    zero_based_mutation_site = mutation_utils.get_zero_based_mutation_site(cln_pdb_file, chain_id, mutation_site)
    print("Row no:{}->{}{}, mutation:{}, 0-based_mutaiton_site:{}".format(i+1, pdb_id, chain_id, mutation, zero_based_mutation_site))

    # generating wild and mutant fasta file
    PDBData.generate_fasta_from_pdb(pdb_id=pdb_id, chain_id=chain_id, input_pdb_filepath=cln_pdb_file, save_as_fasta=True, output_fasta_dir=fastas_dir)
    PDBData.create_mutant_fasta_file(wild_fasta_file=wild_fasta_file, mutant_fasta_file=mutant_fasta_file, zero_based_mutation_site=zero_based_mutation_site, mutant_residue=mutant_residue, mutation=mutation)
    
    print()
    if i+1 == n_rows_to_skip+n_rows_to_evalutate: 
        break
