import pyreadstat

sav_file =  'classify_dino/cifar10/svm_model_grid_search.sav'

with pyreadstat.read_sav(sav_file) as f:
    data, meta = f
    print(data.head())  # print the first few rows of the data
    print(meta)  # print the metadata (e.g., variable names, labels)