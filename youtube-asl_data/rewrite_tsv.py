import pandas as pd
import random
import numpy as np
import math
import sys

def rewrite_tsv(n, tsv_file, output_path):
    seed = 123
    np.random.seed(seed)

    output_file = output_path + '/new_youtube-asl_v1_' + str(n) + '.tsv' 
    test_split = 0.1
    valid_split = 0.1

    df = pd.read_csv(tsv_file, sep='\t')
    df['split'] = 'train'
    
    lst = list(range(len(df)))
    test_len = math.floor(test_split*len(df))
    valid_len = math.floor(valid_split*len(df))
    n = test_len + valid_len
    
    rows = random.sample(lst, n)
    test_rows = rows[:test_len]
    valid_rows = rows[test_len:]
    df.loc[test_rows, 'split'] = 'test'
    df.loc[valid_rows, 'split'] = 'valid'

    print("total length: ", len(df))
    print("test length: ", len(df[df['split']=='test']))
    print("train length: ", len(df[df['split']=='train']))
    print("valid length: ", len(df[df['split']=='valid']))

    df.to_csv(output_file, sep='\t', index=False)

arg = sys.argv[1]
rewrite_tsv(arg, '/home/grt/youtube-asl_data/data/tsv_files/youtube-asl_v1.tsv','/home/grt/youtube-asl_data/data/tsv_files')