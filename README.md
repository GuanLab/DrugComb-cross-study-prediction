# DrugComb Cross dataset prediction model
![](https://github.com/GuanLab/DrugComb-cross-study-prediction/blob/main/Figure1-01.png)
Cross-study prediction of drug combination treatment response

for the dataset used in this study, please refer to [DrugComb](https://drugcomb.fimm.fi/)
## Dependencies

* [python (3.6.5)](https://www.python.org)
* [LightGBM (2.3.2)](https://lightgbm.readthedocs.io/en/latest/index.html)

## Obtaining training dataset

Download the drug combination screening dataset from [DrugComb](https://drugcomb.fimm.fi/) data portal: https://drugcomb.org/download/
and put it under a new directory `./dataset`

## QC

```
QC.ipynb
```

## Split training and test set by study name

```
python split_train_test.py
```
## Construct feature set

```
python preprocess_feature.py
```

## Build model
You can start training model simply by executing the following bash file: 
```
sh bash.sh
```
which will train 20 different models with different feature combinations.

You can also refer to `./master` and run `python main.py -h`

```
usage: main.py [-h] [-f FEATURES [FEATURES ...]]

Build Drugcomb drug combination prediction machine learning models across studies.

optional arguments:
  -h, --help            show this help message and exit
  -f FEATURES [FEATURES ...], --features FEATURES [FEATURES ...]
                         Features selected for model, including:
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
```

this will generate results, save in a new folder `./results`

## Performance visualization
```
demo_results.ipynb
```

## Reference
Zhang, H., Wang, Z., Nan, Y. et al. Harmonizing across datasets to improve the transferability of drug combination prediction. Commun Biol 6, 397 (2023). https://doi.org/10.1038/s42003-023-04783-5

