import sys
sys.path.append("../mutation_analysis_by_DBT")

from objects.PDBData import PDBData
import pickle
  

def get_row_items(row):
    return row["pdb_id"].lower()[:4], row["chain_id"], row["mutation"], int(row["mutation_site"]), row["wild_residue"], row["mutant_residue"], row["ddG"]

def get_zero_based_mutation_site(cln_pdb_file, chain_id, mutation_site):
    pdbdata = PDBData(pdb_dir="data/pdbs_clean/")
    residue_ids_dict = pdbdata.get_residue_ids_dict(pdb_file=cln_pdb_file, chain_id=chain_id)
    return residue_ids_dict.get(mutation_site)

def save_as_pickle(data, path):
    with open(path, "wb") as f: pickle.dump(data, f)

def load_pickle(path):
    with open(path, "rb") as f: return pickle.load(f)