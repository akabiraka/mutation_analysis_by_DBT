# Mutation analysis by DBT


#### Data preparation

* To download and clean PDB files, generate wild and mutant fasta file:
  * `python data_generators/download_pdb_and_gen_fasta.py`
* To encode sequence as blosum50:
  * `python data_generators/encode_seqs.py`


#### Analysis

* To see the data distribution:
  * `python analyzers/data.py`
