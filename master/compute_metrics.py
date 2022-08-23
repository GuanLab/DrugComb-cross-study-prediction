import numpy as np
import pandas as pd
from common import *


all_study = ["ALMANAC", "ONEIL", "FORCINA", "Mathews"]
# save performance

df_out= {'score':[], 'dataset_train':[], 'dataset_test':[], 'rmse':[], 'ci2.5_rmse':[], 'ci97.5_rmse':[], 'ci25_rmse':[], 'ci75_rmse':[]}
for score in ['css', 'synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'S']:
    for train_study in all_study:
        for test_study in all_study:
            out_p = './results/'+train_study+'/'+score+'/'+test_study+'/'
            gs = np.load(out_p+'gs.npy')
            pred = np.load(out_p+'pred.npy')
            #mb, lb, ub = boostrap_ci(pred, gs, 0.95)
            #_, ci25, ci75 = boostrap_ci(pred, gs, 0.5)
            #print("train set %s; test set %s; for %s score: %.4f[%.4f, %.4f]" % (train_study, test_study, score, mb, lb, ub))
            mb_rmse, lb_rmse, ub_rmse = boostrap_ci(pred, gs, 0.95, 'rmse')
            _, ci25_rmse, ci75_rmse = boostrap_ci(pred, gs, 0.5, 'rmse')
            print("train set %s; test set %s; for %s score: %.4f[%.4f, %.4f]" % (train_study, test_study, score, mb_rmse, lb_rmse, ub_rmse))
            # output format
            df_out['score'].append(score)
            df_out['dataset_train'].append(train_study)
            df_out['dataset_test'].append(test_study)
            #df_out['pearsonr'].append(mb)
            #df_out['ci2.5'].append(lb)
            #df_out['ci97.5'].append(ub)
            #df_out['ci25'].append(ci25)
            #df_out['ci75'].append(ci75)
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
            #mb, lb, ub = boostrap_ci(pred, gs, 0.95)
            #_, ci25, ci75 = boostrap_ci(pred, gs, 0.5)
            #print("train set %s; test set %s; for %s score: %.4f[%.4f, %.4f]" % ('_'.join(sorted(train_study)), '_'.join(sorted(test_study)), score, mb, lb, ub))
            
            mb_rmse, lb_rmse, ub_rmse = boostrap_ci(pred, gs, 0.95, 'rmse')
            _, ci25_rmse, ci75_rmse = boostrap_ci(pred, gs, 0.5, 'rmse')
            print("train set %s; test set %s; for %s score: %.4f[%.4f, %.4f]" % ('_'.join(sorted(train_study)), '_'.join(sorted(test_study)), score, mb_rmse, lb_rmse, ub_rmse))
            # output format
            df_out['score'].append(score)
            df_out['dataset_train'].append('_'.join(sorted(train_study)))
            df_out['dataset_test'].append('_'.join(sorted(test_study)))
            #df_out['pearsonr'].append(mb)
            #df_out['ci2.5'].append(lb)
            #df_out['ci97.5'].append(ub)
            #df_out['ci25'].append(ci25)
            #df_out['ci75'].append(ci75)
            df_out['rmse'].append(mb_rmse)
            df_out['ci2.5_rmse'].append(lb_rmse)
            df_out['ci97.5_rmse'].append(ub_rmse)
            df_out['ci25_rmse'].append(ci25_rmse)
            df_out['ci75_rmse'].append(ci75_rmse)

df_out = pd.DataFrame.from_dict(df_out)
df_out.to_csv('./results/performances_rmse.csv', index = False)