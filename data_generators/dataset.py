import sys
sys.path.append("../mutation_analysis_by_DBT")

import numpy as np
import pandas as pd

def data_setup(data_path, batch_size):
    data =  pd.read_csv(data_path, header=None)
    print(data.shape)

    split_num=int(len(data) / batch_size)
    batched_data=np.array_split(data, split_num)
    print("Total batches: " + str(len(batched_data)))
    return batched_data
    
    
def get_batched_data(data_path, batch_size):
    wild_col, mut_col, ddg, label_col = "wild", "mutant", "ddg", "label"
    data =  pd.read_csv(data_path)

    stabilizing = data[data[label_col]=="stabilizing"]
    destabilizing = data[data[label_col]=="destabilizing"]
    stab_n_rows, destab_n_rows = stabilizing.shape[0], destabilizing.shape[0]
    print("#stabilizing: {}, #destabilizing: {}".format(stab_n_rows, destab_n_rows))

    sample_size=int(batch_size/2)
    batched_data = []
    while destab_n_rows > 0:
        restart_stab_sampling_flag=False
        # if stabilizing rows < sample_size, select downsized batch
        if stab_n_rows<sample_size: 
            sample_size=stab_n_rows 
            restart_stab_sampling_flag=True
        # if destabilizing rows < sample_size, select downsized batch
        elif destab_n_rows<sample_size:
            sample_size=destab_n_rows
        else: sample_size=int(batch_size/2)
        
        # random sampling from stabilizing and destabilizing
        stab_sampled = stabilizing.sample(n=sample_size)
        destab_sampled = destabilizing.sample(n=sample_size)

        # shuffle the sampled data
        sampled = pd.concat([stab_sampled, destab_sampled])
        shuffled = sampled.sample(frac=1).reset_index(drop=True)
        
        # without replacement: remove the sampled rows 
        stabilizing = stabilizing.drop(stab_sampled.index)
        destabilizing = destabilizing.drop(destab_sampled.index)

        batched_data.append(shuffled)
        stab_n_rows, destab_n_rows = stabilizing.shape[0], destabilizing.shape[0]
        
        # if there is not stabilizing mutations, restart 
        if restart_stab_sampling_flag: 
            stabilizing = data[data[label_col]=="stabilizing"]
            stab_n_rows, destab_n_rows = stabilizing.shape[0], destabilizing.shape[0]
            # break
    print("Total batches: " + str(len(batched_data)))
    return batched_data


# sample usage
# batched_data = get_batched_data("data/datasets/train.csv", batch_size=32)
# print("Total batches: " + str(len(batched_data)))   
# for i, batch_df in enumerate(batched_data):
#     print(batch_df.shape)
#     print(batch_df.head())
#     break