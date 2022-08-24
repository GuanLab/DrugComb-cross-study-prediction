# M0: baseline
cp -r master M0
cd M0
python main.py -f drug_categorical cell_line_categorical
cd ..

# M1: fancy_baseline
cp -r master M1
cd M1
python main.py -f drug_categorical cell_line_categorical cancer_gene_expression chemical_structure
cd ..

# M2: monotherapy only
cp -r master M2
cd M2
python main.py -f monotherapy_ic50
cd ..

# M3: monotherapy only
cp -r master M3
cd M3
python main.py -f monotherapy_ri
cd ..

# M4: monotherapy only
cp -r master M4
cd M4
python main.py -f monotherapy_ri monotherapy_ic50
cd ..

# M5: drc baseline only
cp -r master M5
cd M5
python main.py -f drc_baseline
cd ..

# M6: drc interpolation
cp -r master M6
cd M6
python main.py -f drc_intp_linear
cd ..

# M7: drc interpolation
cp -r master M7
cd M7
python main.py -f drc_intp_lagrange
cd ..

# M8: drc interpolation
cp -r master M8
cd M8
python main.py -f drc_intp_4PL
cd ..

# M9: monotherapy + drc interpolation
cp -r master M9
cd M9
python main.py -f monotherapy_ic50 monotherapy_ri drc_intp_4PL
cd ..

# M9: M2+ monotherapyic50/ri
cp -r master M10
cd M10
python main.py -f drug_categorical cell_line_categorical cancer_gene_expression chemical_structure monotherapy_ic50 monotherapy_ri
cd ..

# M10: M3+drc_baseline
cp -r master M11
cd M11
python main.py -f drug_categorical cell_line_categorical cancer_gene_expression chemical_structure monotherapy_ic50 monotherapy_ri drc_baseline
cd ..

# M11: M3+drc_intp_linear
cp -r master M12
cd M12
python main.py -f drug_categorical cell_line_categorical cancer_gene_expression chemical_structure monotherapy_ic50 monotherapy_ri drc_intp_linear
cd ..

# M12: M3+drc_intp_lagrange
cp -r master M13
cd M13
python main.py -f drug_categorical cell_line_categorical cancer_gene_expression chemical_structure monotherapy_ic50 monotherapy_ri drc_intp_lagrange
cd ..

# M13: M3+drc_intp_linear
cp -r master M14
cd M14
python main.py -f drug_categorical cell_line_categorical cancer_gene_expression chemical_structure monotherapy_ic50 monotherapy_ri drc_intp_4PL
cd ..

# M13: M3+drc_intp_linear
cp -r master M15
cd M15
python main.py -f drug_categorical cell_line_categorical cancer_gene_expression chemical_structure monotherapy_ic50 monotherapy_ri drc_intp_linear drc_intp_lagrange drc_intp_4PL
cd ..

#M16: drc interpolation
cp -r master M16
cd M16
python main.py -f drc_intp_linear drc_intp_lagrange drc_intp_4PL
cd ..

cp -r master M17
cd M17
python main.py -f monotherapy_ic50 monotherapy_ri drc_intp_linear drc_intp_lagrange drc_intp_4PL
cd ..

#M16: drc interpolation
cp -r master M18
cd M18
python main.py -f drc_baseline drc_intp_linear drc_intp_lagrange drc_intp_4PL
cd ..

#M16: drc interpolation
cp -r master M19
cd M19
python main.py -f monotherapy_ic50 monotherapy_ri drc_baseline drc_intp_linear drc_intp_lagrange drc_intp_4PL
cd ..

