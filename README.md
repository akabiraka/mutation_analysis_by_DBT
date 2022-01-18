# Mutation analysis by DBT

#### Data preparation

* To download and clean PDB files, generate wild and mutant fasta file:
  * `python data_generators/download_pdb_and_gen_fasta.py`
* To encode sequence as blosum50:
  * `python data_generators/encode_seqs.py`
* To create batch data, randomly sample without replacement from both class of batch size:
  * `python data_generators/dataset.py`
  * It is used at training.

#### Training

* To train Net1 model:
  * `python models/net1_trainer.py`
  * It has two parts: randomly initialize model to the best pearson and train from there.

#### Analysis

* To see the data distribution:
  * `python analyzers/data.py`
