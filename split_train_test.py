#!usr/env/bin python3
"""Preprocess the training data
author: @rayezh
"""

from xml.dom import ValidationErr
import pandas as pd
import os, pickle
from sklearn.model_selection import train_test_split, KFold


df_all = pd.read_csv('./dataset/cleaned_summary.csv')
df_all = df_all.dropna(subset = ["css","synergy_zip","synergy_bliss","synergy_loewe","synergy_hsa", "S"])

# split train and test by drug_col, drug_row and cell_line_name
def split_train_test(df, all_study):
    '''
    #test set 1: test within indications
    *split by cell lines(balanced by indications)
    '''

    p1 = './dataset_split'
    os.makedirs(p1, exist_ok = True)
    
    # 1v1
    for study_name in all_study:
        print(study_name)
        p2 = p1+'/'+study_name
        os.makedirs(p2, exist_ok = True) 
        # test_train_split_by_combs
        df_tmp = df.loc[df['study_name'] == study_name,:]
        df_tmp.to_csv(p2+'/all.csv', index = False) 
        df_tmp['comb'] =  ['_'.join(sorted([r.drug_col,r.drug_row])+[r.cell_line_name]) for _,r in df_tmp.iterrows()]
        all_comb = sorted(set(df_tmp.comb))
        print("all experiments:", df_tmp.shape[0])
        print("all combination treatment-cell line combinations:", len(all_comb))

        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        i = 0
        #5-fold cross validation
        for train, test in kf.split(all_comb):
            p3 = p2+'/fold_'+str(i)
            os.makedirs(p3, exist_ok = True)
            i+=1
            train_f = [all_comb[j] for j in train]
            test_f = [all_comb[j] for j in test]
            train_df = df_tmp[df_tmp['comb'].isin(train_f)]
            test_df = df_tmp[df_tmp['comb'].isin(test_f)]
            train_df.to_csv(p3+'/train.csv', index = False)
            test_df.to_csv(p3+'/test.csv', index = False)
    # 3v1
    for study_name in all_study:
        train_study = [s for s in all_study if s != study_name]
        print(train_study)
        # train: 3 study
        p2 = p1+'/'+'_'.join(sorted(train_study))
        os.makedirs(p2, exist_ok = True)
        # test_train_split_by_combs
        df_tmp = df.loc[df['study_name'].isin(train_study),:]
        df_tmp.to_csv(p2+'/all.csv', index = False) 
        df_tmp['comb'] =  ['_'.join(sorted([r.drug_col,r.drug_row])+[r.cell_line_name]) for _,r in df_tmp.iterrows()]
        all_comb = sorted(set(df_tmp.comb))
        print("all experiments:", df_tmp.shape[0])
        print("all combination treatment-cell line combinations:", len(all_comb))

        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        i = 0
        #5-fold cross validation
        for train, test in kf.split(all_comb):
            p3 = p2+'/fold_'+str(i)
            os.makedirs(p3, exist_ok = True)
            i+=1
            train_f = [all_comb[j] for j in train]
            test_f = [all_comb[j] for j in test]
            train_df = df_tmp[df_tmp['comb'].isin(train_f)]
            test_df = df_tmp[df_tmp['comb'].isin(test_f)]
            train_df.to_csv(p3+'/train.csv', index = False)
            test_df.to_csv(p3+'/test.csv', index = False) 
        


# inter-study train and split
all_study = ["ONEIL", "ALMANAC", "FORCINA", "Mathews"]
split_train_test(df_all, all_study)




