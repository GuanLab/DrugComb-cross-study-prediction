import pandas as pd
import numpy as np
import pickle
import os

class feature_construction:
    ''' 
    params:

    yield:
        features: np.array
        feature_names: list

    '''
    def __init__(self, features):
        ''' Load dict for selected features
        '''
        # key is feature
        self.f_dict = {}
        self.f_name = {}
        self.features = features


        for f in self.features:
            feature = pickle.load(open('../features/'+f+'_feature.pkl', 'rb'))
            self.f_dict.update({f:feature})
            if f in ['cancer_gene_expression', 'chemical_structure', 'drc_baseline', 'drc_intp_linear', 'drc_intp_lagrange', 'drc_intp_4PL']:
                feature_name = pickle.load(open('../features/'+f+'_name.pkl', 'rb'))
            else:
                feature_name = [f]
            if f in ['drug_categorical','chemical_structure', 'monotherapy_ic50','monotherapy_ri','drc_baseline','drc_intp_linear', 'drc_intp_lagrange','drc_intp_4PL']:
                r = [i+'_row' for i in feature_name]
                c = [i+'_col' for i in feature_name]
                feature_name = r+c
            self.f_name.update({f:feature_name})
        
    def make_feature(self, df):
        df['block_id'] = df['block_id'].astype(str)

        # extract information
        cell = df['cell_line_name']
        drug_row = df['drug_row']
        drug_col = df['drug_col']
        bid_drug_row = df['block_id']+'|'+df['drug_row']
        bid_drug_col = df['block_id']+'|'+df['drug_col']
        
        x_all = []
        name_all = []
        for f in self.features:
            if f in ['cell_line_categorical', 'cancer_gene_expression']:
                x = np.array([self.f_dict[f][i] for i in cell])
                #print(x.shape)
                x_all.append(x)
            elif f in ['drug_categorical', 'chemical_structure']:
                x = np.array([self.f_dict[f][i] for i in drug_row])
                x_all.append(x)
                x = np.array([self.f_dict[f][i] for i in drug_col])
                x_all.append(x)
                #print(x.shape)
            elif f in ['monotherapy_ic50', 'monotherapy_ri', 'drc_baseline', 'drc_intp_linear', 'drc_intp_lagrange', 'drc_intp_4PL']:
                x = np.array([self.f_dict[f][i] for i in bid_drug_row])
                x_all.append(x)
                x = np.array([self.f_dict[f][i] for i in bid_drug_col])
                x_all.append(x)
                #print(x.shape)
            
            name_all.extend(self.f_name[f])
        
        #print(x_all)
        x_all = np.concatenate(x_all, axis = 1)
        print('feature space shape:',x_all.shape)
        #print(name_all)
        
        return x_all, name_all

