import pandas as pd
import pickle, os
import numpy as np
import numpy.ma as ma
from scipy import sparse 
from feature_construction import feature_construction
from models import *
from utils import *


def intra_study_cv(study, d2f):
    '''
    yields:
        model: pkl
        features: npy
        feature_names: pkl
        eva: tsv
    '''
    print("Train model on "+'_'.join(sorted(study))+" ...")
    # IN
    p1 = '../dataset_split/'+'_'.join(sorted(study))
    # OUT
    out_p1 = './results/'+'_'.join(sorted(study))
    os.makedirs(out_p1, exist_ok=True)

    # feature_construction: dataset to features
    

    for i in range(5):
        # input path
        p2 = p1+ '/fold_'+str(i)

        # training data
        df_train = pd.read_csv(p2+'/train.csv')
        # for training data switch the order between drug_row and drug_col fpr augmentation
        tmp = df_train.rename(columns = {'drug_row': 'drug_col', 'drug_col':'drug_row', 'ic50_row': 'ic50_col', 'ic50_col': 'ic50_row', 'ri_row': 'ri_col', 'ri_col': 'ri_row'})
        df_train = pd.concat([df_train, tmp])

        # test data
        df_test = pd.read_csv(p2+'/test.csv')

        x_train, f_name = d2f.make_feature(df_train[['block_id', 'drug_row', 'drug_col', 'cell_line_name']])
        

        x_test, _ = d2f.make_feature(df_test[['block_id', 'drug_row', 'drug_col', 'cell_line_name']])
        # prediction
        for score in ['css', 'synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'S']:

            out_p2 = out_p1+'/'+score+'/'
            os.makedirs(out_p2, exist_ok=True)

            y_train = df_train[score]
            regressor = train_LightGBM(x_train, y_train, f_name)

            # save model
            print('saving model ...')
            pickle.dump(regressor, open(out_p2+'model_'+str(i)+'.pkl', 'wb'))

            #prediction
            y_gs_test = df_test[score]
            y_pred_test = regressor.predict(x_test)
            
            out_p3 = out_p2+'_'.join(sorted(study))+'/'
            os.makedirs(out_p3, exist_ok=True)
            # pearsonr, rmse
            cor = ma.corrcoef(ma.masked_invalid(y_pred_test), ma.masked_invalid(y_gs_test))[0,1]
            print("Prediction-gold standard's correlation on fold "+str(i)+" test set for "+score+" score: ", cor)

            np.save(out_p3+'gs_'+str(i)+'.npy', y_gs_test)
            np.save(out_p3+'pred_'+str(i)+'.npy', y_pred_test)
            
            sparse.save_npz(out_p3+'x_test_'+str(i)+'.npz', sparse.csr_matrix(x_test))
            pickle.dump(f_name, open(out_p3+'feature_names.pkl', 'wb'))
    
    #ensemble
    for score in ['css', 'synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'S']:
        out_p2 = out_p1+'/'+score+'/'+'_'.join(sorted(study))+'/'
        gs0 = np.load(out_p2+'gs_0.npy')
        gs1 = np.load(out_p2+'gs_1.npy')
        gs2 = np.load(out_p2+'gs_2.npy')
        gs3 = np.load(out_p2+'gs_3.npy')
        gs4 = np.load(out_p2+'gs_4.npy')
        gs = np.concatenate([gs0, gs1, gs2, gs3, gs4])
        pred0 = np.load(out_p2+'pred_0.npy')
        pred1 = np.load(out_p2+'pred_1.npy')
        pred2 = np.load(out_p2+'pred_2.npy')
        pred3 = np.load(out_p2+'pred_3.npy')
        pred4 = np.load(out_p2+'pred_4.npy')
        pred = np.concatenate([pred0, pred1, pred2, pred3, pred4])
        np.save(out_p2+'gs.npy', gs)
        np.save(out_p2+'pred.npy', pred)
        cor = ma.corrcoef(ma.masked_invalid(y_pred_test), ma.masked_invalid(y_gs_test))[0,1]
        print("Correlation on all test set for %s score: %.4f" % (score, cor))



def cross_study_ensemble(train_study, d2f):
    '''
    yields:
        features: npy
        feature_names: pkl
        eva: tsv
    '''
    test_study = [s for s in ["ALMANAC", "ONEIL", "FORCINA", "Mathews"] if s not in train_study]

    for score in ['css', 'synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'S']:
        # load model of train study
        p1 = './results/'+'_'.join(sorted(train_study))+'/'+score+'/'
        model0 = pickle.load(open(p1+'model_0.pkl', 'rb')) # train_study
        model1 = pickle.load(open(p1+'model_1.pkl', 'rb'))
        model2 = pickle.load(open(p1+'model_2.pkl', 'rb'))
        model3 = pickle.load(open(p1+'model_3.pkl', 'rb'))
        model4 = pickle.load(open(p1+'model_4.pkl', 'rb'))
        
        for study in test_study:

            print("Applying "+'_'.join(sorted(train_study))+" model on "+study+" ...")
            
            df_test = pd.read_csv('../dataset_split/'+study+'/all.csv')
            x_test, f_name = d2f.make_feature(df_test[['block_id', 'drug_row', 'drug_col', 'cell_line_name']])
            # out_path
            out_p1 = p1+study+'/'
            os.makedirs(out_p1, exist_ok =True)
            
            # prediction
            gs_test = df_test[score]
            pred0 = model0.predict(x_test)
            pred1 = model0.predict(x_test)
            pred2 = model0.predict(x_test)
            pred3 = model0.predict(x_test)
            pred4 = model0.predict(x_test)
            
            # ensemble
            pred_test = (pred0+pred1+pred2+pred3+pred4)/5
            
            # evaludation
            cor = ma.corrcoef(ma.masked_invalid(pred_test), ma.masked_invalid(gs_test))[0,1]
            print("Prediction-gold standard's correlation on test set for "+score+" score: ", cor)

            np.save(out_p1+'gs.npy', gs_test)
            np.save(out_p1+'pred.npy', pred_test)

            sparse.save_npz(out_p1+'x_test.npz', sparse.csr_matrix(x_test))
            pickle.dump(f_name, open(out_p1+'feature_names.pkl', 'wb'))

