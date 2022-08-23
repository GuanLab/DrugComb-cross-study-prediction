# DrugComb Cross dataset prediction model

# dataset:
    * summary.csv: combination response; training set
    * concentration: dose-response curve of all samples. 
    * cancer_genes.tsv #TODO: find reference
    * drug.csv: chemical structure
    * drug_target.csv: # find reference; update this
    * cell_line.csv: cell line information
    * source.csv: which study/assay used
    * disease.csv: disease_id annotation in cell_line.csv
    * tissue.csv: tissue_id annotation in cell_line.csv



# pipeline:
    * QC.ipynb
    * split_train_test.py
    * preprocess_feature.py
    * bash.sh
    * demo_results.ipynb 

running:
bash 1 2 3 4 5
# DrugComb-cross-study-combination-prediction
# DrugComb-cross-study-prediction
