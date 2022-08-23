#! usr/bin/env python3
"""
author: @rayezh

train and test from cross validation
"""
import pandas as pd
import os, sys
import argparse, textwrap
import pickle
from common import *


def main():
    parser = argparse.ArgumentParser(description = "Build Drugcomb drug combination prediction machine learning models across studies.",
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-f', '--features', nargs = '+',
        type = str, 
        default = ['drug_categorical', 'cell_line_categorical'],
        help = ''' Features selected for model, including:
        drug_categorical;
        cell_line_categorical;
        cancer_gene_expression;
        chemical_structure;
        monotherapy_ri;
        monotherapy_ic50;
        drc_baseline;
        drc_intp_linear;
        drc_intp_lagrange;
        drc_intp_4PL;
        (default = ['drug_categorical', 'cell_line_categorical']
        ''')
    args = parser.parse_args()
    print(args.features)
    opts = vars(args)
    run(**opts)


def run(features):

    # feature_construction: dataset to features
    d2f = feature_construction(features)
    all_study = ["ALMANAC", "ONEIL", "FORCINA", "Mathews"]

    os.makedirs('./results/', exist_ok=True)
    
    # 1v1:
    for s in all_study:
        intra_study_cv([s], d2f)
        cross_study_ensemble([s], d2f)

    # 3v1
    for s in all_study:
        new_s = [i for i in all_study if i != s]
        intra_study_cv(new_s, d2f)
        cross_study_ensemble(new_s, d2f)

    # save performance

    df_out= {'score':[], 'dataset_train':[], 'dataset_test':[], 'pearsonr':[], 'ci2.5':[], 'ci97.5':[], 'ci25':[], 'ci75':[], 'rmse':[], 'ci2.5_rmse':[], 'ci97.5_rmse':[], 'ci25_rmse':[], 'ci75_rmse':[]}
    for score in ['css', 'synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'S']:
        for train_study in all_study:
            for test_study in all_study:
                out_p = './results/'+train_study+'/'+score+'/'+test_study+'/'
                gs = np.load(out_p+'gs.npy')
                pred = np.load(out_p+'pred.npy')
                mb, lb, ub = boostrap_ci(pred, gs, 0.95)
                _, ci25, ci75 = boostrap_ci(pred, gs, 0.5)
                print("train set %s; test set %s; for %s score: %.4f[%.4f, %.4f]" % (train_study, test_study, score, mb, lb, ub))    
                mb_rmse, lb_rmse, ub_rmse = boostrap_ci(pred, gs, 0.95, 'rmse')
                _, ci25_rmse, ci75_rmse = boostrap_ci(pred, gs, 0.5, 'rmse')
                print("train set %s; test set %s; for %s score: %.4f[%.4f, %.4f]" % (train_study, test_study, score, mb_rmse, lb_rmse, ub_rmse))
            
                # output format
                df_out['score'].append(score)
                df_out['dataset_train'].append(train_study)
                df_out['dataset_test'].append(test_study)
                df_out['rmse'].append(mb_rmse)
                df_out['ci2.5_rmse'].append(lb_rmse)
                df_out['ci97.5_rmse'].append(ub_rmse)
                df_out['ci25_rmse'].append(ci25_rmse)
                df_out['ci75_rmse'].append(ci75_rmse)
        for train_study in [["ALMANAC", "ONEIL", "FORCINA"], ["ALMANAC", "ONEIL", "Mathews"], ["ALMANAC", "FORCINA", "Mathews"], ["ONEIL", "FORCINA", "Mathews"]]:
            t_study = [[s] for s in ["ALMANAC", "ONEIL", "FORCINA", "Mathews"] if s not in train_study]+[train_study]
            for test_study in t_study:
                out_p = './results/'+'_'.join(sorted(train_study))+'/'+score+'/'+'_'.join(sorted(test_study))+'/'
                gs = np.load(out_p+'gs.npy')
                pred = np.load(out_p+'pred.npy')
                mb, lb, ub = boostrap_ci(pred, gs, 0.95)
                _, ci25, ci75 = boostrap_ci(pred, gs, 0.5)
                print("train set %s; test set %s; for %s score: %.4f[%.4f, %.4f]" % ('_'.join(sorted(train_study)), '_'.join(sorted(test_study)), score, mb, lb, ub))

                mb_rmse, lb_rmse, ub_rmse = boostrap_ci(pred, gs, 0.95, 'rmse')
                _, ci25_rmse, ci75_rmse = boostrap_ci(pred, gs, 0.5, 'rmse')
                print("train set %s; test set %s; for %s score: %.4f[%.4f, %.4f]" % ('_'.join(sorted(train_study)), '_'.join(sorted(test_study)), score, mb_rmse, lb_rmse, ub_rmse))
                # output format
                df_out['score'].append(score)
                df_out['dataset_train'].append('_'.join(sorted(train_study)))
                df_out['dataset_test'].append('_'.join(sorted(test_study)))
                df_out['pearsonr'].append(mb)
                df_out['ci2.5'].append(lb)
                df_out['ci97.5'].append(ub)
                df_out['ci25'].append(ci25)
                df_out['ci75'].append(ci75)
                df_out['rmse'].append(mb_rmse)
                df_out['ci2.5_rmse'].append(lb_rmse)
                df_out['ci97.5_rmse'].append(ub_rmse)
                df_out['ci25_rmse'].append(ci25_rmse)
                df_out['ci75_rmse'].append(ci75_rmse)

    df_out = pd.DataFrame.from_dict(df_out)
    df_out['features'] = '+'.join(features)
    df_out.to_csv('./results/performances.csv', index = False)


if __name__ == "__main__":
    main()





