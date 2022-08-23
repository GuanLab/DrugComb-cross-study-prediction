import pandas as pd
import numpy as np
import os, pickle
from scipy import sparse
from glob import glob 
import shap
import matplotlib.pyplot as plt


def shap_analysis(path):
    '''
    '''
    # model
    print("Carrying out SHAP analysis on the test set...")
    reg0 = pickle.load(open(path+'/model_0.pkl', 'rb'))
    reg1 = pickle.load(open(path+'/model_1.pkl', 'rb'))
    reg2 = pickle.load(open(path+'/model_2.pkl', 'rb'))
    reg3 = pickle.load(open(path+'/model_3.pkl', 'rb'))
    reg4 = pickle.load(open(path+'/model_4.pkl', 'rb'))
    
    exp0 = shap.TreeExplainer(reg0)
    exp1 = shap.TreeExplainer(reg1)
    exp2 = shap.TreeExplainer(reg2)
    exp3 = shap.TreeExplainer(reg3)
    exp4 = shap.TreeExplainer(reg4)

    # feature set
    path_1 = glob(path+'/*')
    path_1 = [i for i in path_1 if not i.endswith('.pkl')]
    for p in path_1:
        print(p)
        if p.split('/')[2] == p.split('/')[4]:
            pass
        else:
            x =  sparse.csr_matrix.toarray(sparse.load_npz(p+'/x_test.npz'))
            feature_names = pickle.load(open(p+'/feature_names.pkl', 'rb'))
            print(len(feature_names))
            new_feature_names = [i.replace('_row','') for i in feature_names]
            new_feature_names = [i.replace('_col','') for i in new_feature_names]
            x = pd.DataFrame(x, columns=new_feature_names)


            shap_v0 = exp0(x)
            shap_v1 = exp1(x)
            shap_v2 = exp2(x)
            shap_v3 = exp3(x)
            shap_v4 = exp4(x)
            shap_v = (shap_v0+shap_v1+shap_v2+shap_v3+shap_v4)/5

            expval0= exp0.expected_value
            #shap_v0 = shap.TreeExplainer(reg0).expected_value
            #shap_v1 = shap.TreeExplainer(reg1).expected_value
            #shap_v2 = shap.TreeExplainer(reg2).shap_values(x)
            #shap_v3 = shap.TreeExplainer(reg3).shap_values(x)
            #shap_v4 = shap.TreeExplainer(reg4).shap_values(x)
            #shap_v = (shap_v0+shap_v1+shap_v2+shap_v3+shap_v4)/5

            print(shap_v.shape)
            

            shap.summary_plot(shap_v, x, feature_names = new_feature_names, show=False)
            shap_fig = plt.gcf()
            plt.savefig('./shap'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()

            # force
            shap.plots.scatter(shap_v[:,"drc_intp_linear_10"], color=shap_v)
            plt.savefig('./shap_scatter_10'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()

            shap.plots.scatter(shap_v[:,"drc_intp_linear_9"], color=shap_v)
            plt.savefig('./shap_scatter_9'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()

            shap.plots.scatter(shap_v[:,"drc_intp_linear_8"], color=shap_v)
            plt.savefig('./shap_scatter_8'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()

            shap.plots.scatter(shap_v[:,"drc_intp_linear_7"], color=shap_v)
            plt.savefig('./shap_scatter_7'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()

            shap.plots.scatter(shap_v[:,"drc_intp_linear_7"], color=shap_v)
            plt.savefig('./shap_scatter_7'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()

            shap.plots.scatter(shap_v[:,"drc_intp_linear_6"], color=shap_v)
            plt.savefig('./shap_scatter_6'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()

            shap.plots.scatter(shap_v[:,"drc_intp_linear_5"], color=shap_v)
            plt.savefig('./shap_scatter_5'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()

            shap.plots.scatter(shap_v[:,"drc_intp_linear_4"], color=shap_v)
            plt.savefig('./shap_scatter_4'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()

            shap.plots.scatter(shap_v[:,"drc_intp_linear_3"], color=shap_v)
            plt.savefig('./shap_scatter_3'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()

            shap.plots.scatter(shap_v[:,"drc_intp_linear_2"], color=shap_v)
            plt.savefig('./shap_scatter_2'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()

            shap.plots.scatter(shap_v[:,"drc_intp_linear_1"], color=shap_v)
            plt.savefig('./shap_scatter_1'+'_'+p.split('/')[2]+'_'+p.split('/')[3]+'_'+p.split('/')[4]+'.png',bbox_inches='tight',dpi=100)
            plt.close()


def main():

    """
    """
    all_study = ["ALMANAC", "ONEIL", "FORCINA", "Mathews"]

    path = './results/'
    
    # 1v1:
    train_study = glob(path+'*')
    train_study = [i for i in train_study if not i.endswith('.csv')]

    for s in train_study:
        print(s)
        for score in ['css', 'synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'S']:
            path = s+'/'+score
            shap_analysis(path)



if __name__ == "__main__":
    main()