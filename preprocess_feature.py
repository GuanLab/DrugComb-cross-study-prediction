#!usr/bin/python3
import pandas as pd
import numpy as np
import collections
from collections import defaultdict
import os, json, pickle
from tqdm import tqdm
from scipy import stats
from glob import glob
from itertools import combinations
from matplotlib import pyplot
import math
from scipy.interpolate import interp1d
from scipy import interpolate 
"""
# chemical structure required modules
from rdkit import Chem,DataStructs
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdmolops
from pubchempy import *
import openbabel
import pybel

"""

def categorical_features():
    ''' categorical features
    '''
    outpath = './features/'
    os.makedirs(outpath, exist_ok = True)

    df = pd.read_csv('./dataset/cleaned_summary.csv', header = 0)
    # cell line
    cell_line = {c:np.array([i] )for i, c in enumerate(sorted(set(df.cell_line_name)))}
    # drug
    drug = {c:np.array([i]) for i, c in enumerate(sorted(set(list(df.drug_row)+list(df.drug_col))))}

    print("total cell lines:", len(cell_line))
    print("total drugs:", len(drug))

    pickle.dump(cell_line, open(outpath+'cell_line_categorical_feature.pkl', "wb"))
    pickle.dump(drug, open(outpath+'drug_categorical_feature.pkl', "wb"))


def chemical_structure_qc():
    """ QC of chemical structure information and feature preprocess
    Transform SMILE formatted chemical structure into 6 types of Fingerprints
    """

    outpath = './features/'
    os.makedirs(outpath, exist_ok = True)

    df_map = pd.read_csv('./dataset/drug.csv', header = 0)
    df_map = df_map.drop_duplicates(subset = ['dname'])


    all_features = {}
    feature_names = ['MACCS_'+str(idx+1) for idx in range(167)]+['Morgan_'+str(idx+1) for idx in range(1024)]+['RDK_'+str(idx+1)for idx in range(2048)]

    for _,r in tqdm(df_map.iterrows(), total = df_map.shape[0]):
        try:
            print(r['dname'], r['smiles'])
            smile = r['smiles']
            ms = Chem.MolFromSmiles(smile)
            mol = pybel.readstring("smi", smile)
        
            # MACCS features (167*1)
            fp = MACCSkeys.GenMACCSKeys(ms)
            tmp = fp.ToBitString()
            feature_1 = list(map(int, tmp))
            #feature_1 = {'MACCS_'+str(idx+1):x for idx,x in enumerate(feature_1)}

            # Morgan Fingerprints (1024*1)
            fp = AllChem.GetMorganFingerprintAsBitVect(ms,2,nBits=1024)
            tmp = fp.ToBitString()
            feature_2 = list(map(int, tmp))
            #feature_2 = {'Morgan_'+str(idx+1):x for idx,x in enumerate(feature_2)}

            # RDK Fingerprints (2048*1)
            fp = rdmolops.RDKFingerprint(ms)
            tmp = fp.ToBitString()
            feature_3 = list(map(int, tmp))
            #feature_3 = {'RDK_'+str(idx+1):x for idx,x in enumerate(feature_6)}

            the_features = {r['dname']:feature_1+feature_2+feature_3}
            all_features.update(the_features)
        except:
            pass

    pickle.dump(all_features, open(outpath+'chemical_structure_feature.pkl', "wb"))
    pickle.dump(feature_names, open(outpath+'chemical_structure_feature_name.pkl', "wb"))

def cancer_gene_expression():
    '''
    '''
    outpath = './features/'
    os.makedirs(outpath, exist_ok = True)

    df = pd.read_csv('./dataset/processed_CCLE_expression.csv', header = 0)
    
    exp_names = df.columns[:-1].to_list()
    exp_features = {r.cell_line_name:r[:-1].to_list() for _, r in df.iterrows()}
    
    pickle.dump(exp_features, open(outpath+'cancer_gene_expression_feature.pkl', "wb"))
    pickle.dump(exp_names, open(outpath+'cancer_gene_expression_feature_name.pkl', "wb"))


def monotherapy_response():
    '''
    monotherapy features
    ic50
    ri
    (DO NOT include CSS!! it's only for combination and causes information leak)
    '''

    outpath = './features/'
    os.makedirs(outpath, exist_ok = True)
    df= pd.read_csv('./dataset/processed_monotherapy_response.csv')
    # ic50
    ic50_features =dict(zip(df['block_id|drug'].to_list(), [[i] for i in df.ic50]))
    # ri (relative inhibition)
    ri_features = dict(zip(df['block_id|drug'].to_list(), [[i] for i in df.ri]))
    # no CSS because this is only for combination; not applicable yo monotherapy
    pickle.dump(ic50_features, open(outpath+'monotherapy_ic50_feature.pkl', "wb"))
    pickle.dump(ri_features, open(outpath+'monotherapy_ri_feature.pkl', "wb"))



def drc_baseline():
    ''' drc baseline feature; pad the rest with -1
    for example, for original data dose: [1,1,1] response: [2,2,2]
    the drc baseline feature will be [1,1,1, -1, -1, -1, 2,2,2,-1, -1, -1]

    '''

    def pad_list(a):
        ''' pad list to 10 by -1'''
        a = list(a)
        return (a+[-1]*(11-len(a)))[:10]


    outpath = './features/'
    os.makedirs(outpath, exist_ok = True)
    df= pd.read_csv('./dataset/processed_concentration.csv')
    print(df)
    #df = df.loc[df['study_name']=='Mathews',:]
    #print(df)
    df = df.groupby('block_id|drug')[['conc', 'inhibition']].agg(pad_list).reset_index()
    drc_base_names = ['dose_'+str(idx+1) for idx in range(10)]+['response_'+str(idx+1) for idx in range(10)]
    drc_base_features = {r['block_id|drug']:r['conc']+r['inhibition'] for _,r in df.iterrows()}
    print('Finished making DR curve baseline feature!')
    print(drc_base_names)
    print(list(drc_base_features.items())[-1])

    pickle.dump(drc_base_features, open(outpath+'drc_baseline_feature.pkl', "wb"))
    pickle.dump(drc_base_names, open(outpath+'drc_baseline_feature_name.pkl', "wb"))


def drc_intp():
    ''' imputed dr curve by interpolation

    '''
    def new_conc(conc, len_const = 10):
        '''Return the interpolated doses for dose-response curve
        
        params:
        conc: list
            concentration interval
        len_const: int 
            max interval length
        
        yield:
        list length of len_const

        '''
        if len(conc) <= 2: 
            # 0.2962 is the average dilution ratio of all the expriments in both row and column drugs expect those only have two dose levels
            # 0.2962 is the dilutaion ratio used to interpolate the concentration data with only two dose points
            return  10 ** np.linspace(math.log(conc[-1] * ((0.2962) ** 9), 10), math.log(conc[-1], 10), num = len_const)
        else:
            ratio = conc[-2] / conc[-1]
            return 10 ** np.linspace(math.log(conc[1] * ratio, 10), math.log(conc[-1], 10), num = len_const)

    def lagrange_intp(x, new_x, y):
        '''
        params:
        x: dose
        new_x: interpolated dose
        y: response

        return:
        new_y
        '''
        f = interpolate.lagrange(x, y)
        return f(new_x)
    
    def linear_intp(x, new_x, y):
        '''
        params:
        x: dose
        new_x: interpolated dose
        y: response

        return:
        new_y
        '''
        return np.interp(new_x, x, y)
    
    def log_intp(x, new_x, y, ic50):
        '''  4 parameter logistic (4PL) equation approach
        This function is basically a copy of the LL.4 function from the R drc package with

        params:
        x: dose
        new_x: interpolated dose
        y: response
        
        return:
        new_y
        ''' 
        a = ((y[-1] - y[0])/(x[-1] - x[0])) # slope factor
        b = max(y) #max response
        c = min(y) # min response
        return (b+(c-b)/(1+(np.array(new_x)/ic50)**a)).tolist()

    outpath = './features/'
    os.makedirs(outpath, exist_ok = True)
    df= pd.read_csv('./dataset/processed_concentration.csv')
    all_blk = df['block_id|drug'].to_list()
    conc = df.groupby('block_id|drug')['conc'].agg(list).to_dict()
    resp = df.groupby('block_id|drug')['inhibition'].agg(list).to_dict()
    
    # preprocess parameters
    ic50 = pickle.load(open(outpath+'monotherapy_ic50_feature.pkl', "rb"))


    
    # linear
    linear_features = {}
    # lagrange
    lagrange_features = {}
    # 4PL
    log_features = {}

    for i in tqdm(all_blk):
        try:
            new_x = new_conc(conc[i])
            # linear
            new_y = linear_intp(conc[i], new_x, resp[i])
            linear_features.update({i:new_y})
            
            # lagrange
            new_y = lagrange_intp(conc[i], new_x, resp[i])
            lagrange_features.update({i:new_y})
            #  4PL
            new_y = log_intp(conc[i],new_x, resp[i], ic50[i])
            log_features.update({i:new_y})
            
        except:
            print(i)
            print(conc[i])
            print(resp[i])
            print(ic50[i])

    pickle.dump(linear_features, open(outpath+'drc_intp_linear_feature.pkl', "wb"))
    pickle.dump(['drc_intp_linear_'+str(i+1) for i in range(10)], open(outpath+'drc_intp_linear_name.pkl', "wb"))
    
    pickle.dump(lagrange_features, open(outpath+'drc_intp_lagrange_feature.pkl', "wb"))
    pickle.dump(['drc_intp_lagrange_'+str(i+1) for i in range(10)], open(outpath+'drc_intp_lagrange_name.pkl', "wb"))

    pickle.dump(log_features, open(outpath+'drc_intp_4PL_feature.pkl', "wb"))
    pickle.dump(['drc_intp_4PL_'+str(i+1) for i in range(10)], open(outpath+'drc_intp_4PL_name.pkl', "wb"))
    

#categorical_features()
#chemical_structure_qc()
#cancer_gene_expression()
#monotherapy_response()
drc_baseline()
#drc_intp()
