import sys
sys.path.append("../mutation_analysis_by_DBT")

import numpy as np
import re

class SeqEcodeHelper(object):
    def __init__(self, blosum_path="data/blosum50.txt") -> None:
        super().__init__()
        self.AA={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}
        self.blosum_path = blosum_path

    def read_blosum(self):
        blosum = []
        with open(self.blosum_path,"r") as f:
            for line in f:
                blosum.append([(float(i))/10 for i in re.split("\t",line)])
                #The values are rescaled by a factor of 1/10 to facilitate training
        return blosum

    
    def seq_to_blosum(self, seq):
        blosum_matrix = self.read_blosum()
        seqe = []
        for aa in seq:
            seqe.append(blosum_matrix[self.AA[aa]])
        
        blosum_encoded = np.array(seqe, dtype=np.float32)
        return blosum_encoded


    def cut_mutation_region(self, seq, mutation_pos, sub_seq_len=15):
        j=1
        sub_seq_len-=1
        from_index, to_index= 0, len(seq)-1
        while(sub_seq_len>0):
            # pointer going right direction
            if mutation_pos+j in range(len(seq)): 
                to_index=mutation_pos+j
                sub_seq_len-=1
            # pointer going left direction
            if mutation_pos-j in range(len(seq)):
                from_index=mutation_pos-j
                sub_seq_len-=1

            j+=1
            # print(len(seq[from_index:to_index+1]), seq[from_index:to_index+1], from_index, to_index, sub_seq_len)
        return seq[from_index:to_index+1]

    

# sample usage
seq_encode_helper = SeqEcodeHelper()    

seq_encode_helper.seq_to_blosum("ARN")

# blosum_matrix = seq_encode_helper.read_blosum(path="data/blosum50.txt")
# # print(len(blosum_matrix), len(blosum_matrix[0])) # row, col

# seq = "ARN"
# mutation_pos = 3
# seq_encode_helper.cut_mutation_region(seq, mutation_pos, 3)