#run dino.py first, after that you will get train and test in .pth file.

import argparse
import os
import time
from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import utils

def get_file_size_in_kb(file_path):
    file_size_bytes = os.path.getsize(file_path)  # Get file size in bytes
    file_size_kb = file_size_bytes / 1024  # Convert bytes to kilobytes
    return file_size_kb

def calculate_topk_accuracy(output, target, topk=(1, 5)):
    # https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def knn_classify(name_dataset, X_train, y_train, X_test, y_test, dino_dir, _size, act_pca, n_component, svd_solver):
    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=20) 
    #Number of NN to use. 20 is usually working the best. #https://github.com/facebookresearch/dino/blob/main/eval_knn.py
    knn.fit(X_train, y_train.ravel())

    # Save the model
    with open(os.path.join(dino_dir, 'knn_model.sav'), 'wb') as pickle_file:
        pickle.dump(knn, pickle_file)

    # Predict probabilities
    y_pred_prob = knn.predict_proba(X_test)
    y_pred = knn.predict(X_test)

    # Calculate normal accuracy
    accuracy = (accuracy_score(y_test, y_pred))*100

    # Calculate top-1 and top-5 accuracy
    y_test_tensor = torch.tensor(LabelEncoder().fit_transform(y_test))
    y_pred_prob_tensor = torch.tensor(y_pred_prob)
   
    top1_acc, top5_acc = calculate_topk_accuracy(y_pred_prob_tensor, y_test_tensor)

    # Save classification report
    with open(os.path.join(dino_dir, "classification_report_knn.txt"), 'a') as fd:
        fd.write(f'Size: {_size}\n')
        fd.write(f'Accuracy: {accuracy}%\n')
        fd.write(f'Top-1 Accuracy: {top1_acc.item():.2f}%\n')
        fd.write(f'Top-5 Accuracy: {top5_acc.item():.2f}%\n')
        fd.write(f'Classification report: \n{classification_report(y_test, y_pred)}\n')
        fd.write(f'Parameters: \n{knn.get_params()}\n\n\n')

    #combine all results in one csv file
    if act_pca:
        with open(os.path.join(f'classify_pca_dino_svd_solver_{svd_solver}', f"report_pca_dino_knn_{svd_solver}.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{accuracy}%;{top1_acc.item():.2f}%;{top5_acc.item():.2f}%;{n_component};{_size};{time.time()}\n')   
    else:
        with open(os.path.join('classify_dino', "report_dino_knn.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{accuracy}%;{top1_acc.item():.2f}%;{top5_acc.item():.2f}%;{_size};{time.time()}\n') 

def svm_classify(name_dataset, X_train, y_train, X_test, y_test, dino_dir, _size, act_pca, n_component, svd_solver):
    
    # Train SVM model
    clf = svm.SVC(kernel="linear", verbose=False) #default setting, we use it on this research.

    #best configuration
    # clf = svm.SVC(kernel="rbf", verbose=False, C=10, gamma="scale", tol=0.001) #first best setting for cifar10 and eurosat
    # clf = svm.SVC(kernel="linear", verbose=False, C=0.01, gamma="scale", tol=0.001) #second best setting for caltech101
    # clf = svm.SVC(kernel="rbf", verbose=False, C=10, gamma="auto", tol=0.001) #third best setting for cifar100
    # clf = svm.SVC(kernel="linear", verbose=False, C=100, gamma="scale", tol=0.001) #fourth best setting for caltech256

    clf.fit(X_train, y_train.ravel())

    # Save the model
    with open(os.path.join(dino_dir, 'svm_model.sav'), 'wb') as pickle_file:
        pickle.dump(clf, pickle_file)

    # Predict
    y_pred = clf.predict(X_test)
    # Calculate accuracy
    accuracy = (accuracy_score(y_test, y_pred))*100

    # Calculate top-1 and top-5 accuracy
    y_pred_prob = clf.decision_function(X_test)  # Get decision function scores
    y_test_tensor = torch.tensor(LabelEncoder().fit_transform(y_test))
    y_pred_prob_tensor = torch.tensor(y_pred_prob)

    top1_acc, top5_acc = calculate_topk_accuracy(y_pred_prob_tensor, y_test_tensor)

    # Save classification report
    with open(os.path.join(dino_dir, "classification_report_svm.txt"), 'a') as fd:
        fd.write(f'Size: {_size}\n')
        fd.write(f'Accuracy: {accuracy}%\n')
        fd.write(f'Top-1 Accuracy: {top1_acc.item():.2f}%\n')
        fd.write(f'Top-5 Accuracy: {top5_acc.item():.2f}%\n')
        fd.write(f'Classification report: \n{classification_report(y_test, y_pred)}\n')
        fd.write(f'Parameters: \n{clf.get_params()}\n\n\n')
    
    #combine all results in one csv file
    if act_pca:
        with open(os.path.join(f'classify_pca_dino_svd_solver_{svd_solver}', f"report_pca_dino_svm_{svd_solver}.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{accuracy}%;{top1_acc.item():.2f}%;{top5_acc.item():.2f}%;{n_component};{_size};{time.time()}\n')   
    else:
        with open(os.path.join('classify_dino', "report_dino_svm.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{accuracy}%;{top1_acc.item():.2f}%;{top5_acc.item():.2f}%;{_size};{time.time()}\n') 

def main(dataset, act_pca, n_component, pth, svd_solver_args, fp16):

    # Paths to data and results
    name_dataset=dataset
    # src_dir = os.path.join("images/", f"{name_dataset}")
    file_pth = os.path.join(f"{pth}", f"{name_dataset}")

    X_train = torch.load(os.path.join(file_pth,'trainfeat.pth')).cpu().numpy()
    X_test = torch.load(os.path.join(file_pth,'testfeat.pth')).cpu().numpy()
    y_train = torch.load(os.path.join(file_pth,'trainlabels.pth')).numpy()
    y_test = torch.load(os.path.join(file_pth,'testlabels.pth')).numpy()

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    if fp16:    
        print("features use fp16")
        X_train = X_train.astype(np.float16)
        X_test = X_test.astype(np.float16)
    else:
        print("features use fp32")

    if act_pca:
        print(f"PCA-Dino Classifier with n_components = {n_component}")
        
        pca_dino_dir = os.path.join(f"classify_pca_dino_svd_solver_{svd_solver_args}", f"{name_dataset}/{n_component}")
        if not os.path.exists(pca_dino_dir):
            os.makedirs(pca_dino_dir)
        pickle.dump(sc, open(os.path.join(pca_dino_dir, 'standard_scaler.sav'), 'wb'))
        
        # PCA
        if n_component is not None:
            if n_component < 1:
                pca = PCA(n_component, svd_solver=svd_solver_args)
            else:
                pca = PCA(min(n_component, X_train.shape[0], X_train.shape[1]), svd_solver=svd_solver_args)
        else:
            pca = PCA(svd_solver=svd_solver_args,)
        
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test) 

        if fp16:    
            print(".")
            X_train = X_train.astype(np.float16)
            X_test = X_test.astype(np.float16)
        else:
            print(".")

        pickle.dump(pca, open(os.path.join(pca_dino_dir, 'pca_model.sav'), 'wb'))

        with open(os.path.join(pca_dino_dir, 'X_train_pca-dino.npy'),'wb') as npy_file:
            np.save(npy_file, X_train)
        
        with open(os.path.join(pca_dino_dir, 'X_test_pca-dino.npy'),'wb') as npy_file:
            np.save(npy_file, X_test)

        with open(os.path.join(pca_dino_dir, 'y_train_pca-dino.npy'),'wb') as npy_file:
            np.save(npy_file, y_train)
        
        with open(os.path.join(pca_dino_dir, 'y_test_pca-dino.npy'),'wb') as npy_file:
            np.save(npy_file, y_test)

        _size_pca = get_file_size_in_kb(os.path.join(pca_dino_dir, 'X_train_pca-dino.npy'))+get_file_size_in_kb(os.path.join(pca_dino_dir, 'X_test_pca-dino.npy'))
        
        with open(os.path.join(pca_dino_dir, "pca_report.txt"),'a') as fd:
            fd.write(f'Size (kb): {_size_pca}\n')
            fd.write(f'Explained variance: {sum(pca.explained_variance_ratio_)}\n')
            fd.write(f'Number of components: {pca.n_components_}\n')
            fd.write(f'Number of features: {pca.n_features_in_}\n')
            fd.write(f'svd_solver: {svd_solver_args}\n')
            fd.write(f'Parameters: {pca.get_params()}\n')
            fd.write(f'Number of samples: {pca.n_samples_}\n\n\n')

        svm_classify(name_dataset, X_train, y_train, X_test, y_test, pca_dino_dir, _size_pca, act_pca, n_component, svd_solver_args)
        knn_classify(name_dataset, X_train, y_train, X_test, y_test, pca_dino_dir, _size_pca, act_pca, n_component, svd_solver_args)

        print("Classify Done")

    else:
        print("Only Dino Classifiers")  
        dino_dir = os.path.join("classify_dino", f"{name_dataset}")
    
        if not os.path.exists(dino_dir):
            os.makedirs(dino_dir) 
        
        with open(os.path.join(dino_dir, 'X_train-dino.npy'),'wb') as npy_file:
            np.save(npy_file, X_train)
        
        with open(os.path.join(dino_dir, 'X_test-dino.npy'),'wb') as npy_file:
            np.save(npy_file, X_test)

        with open(os.path.join(dino_dir, 'y_train-dino.npy'),'wb') as npy_file:
            np.save(npy_file, y_train)
        
        with open(os.path.join(dino_dir, 'y_test-dino.npy'),'wb') as npy_file:
            np.save(npy_file, y_test)

        _size = get_file_size_in_kb(os.path.join(dino_dir, 'X_train-dino.npy'))+get_file_size_in_kb(os.path.join(dino_dir, 'X_test-dino.npy'))
        svm_classify(name_dataset, X_train, y_train, X_test, y_test, dino_dir, _size, act_pca, n_component, svd_solver_args)
        knn_classify(name_dataset, X_train, y_train, X_test, y_test, dino_dir, _size, act_pca, n_component, svd_solver_args)

        print("Classify done")
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser('PCA-Dino')
    parser.add_argument("--dataset", default="caltech101", type=str, help="""set your actual name of your dataset""")
    parser.add_argument("--act_pca", default=False, type=utils.bool_flag, help="""set True if you want using PCA""")
    parser.add_argument("--n_component", default=20, type=int, help="""using this if you used PCA""")
    parser.add_argument("--load_features", default=None, help="""using this for load you .pth, npy, or pt file to train and test,
                        there are four file which you need first trainfeat, testfeat, trainlabels, and testlabels""")
    parser.add_argument("--svd_solver", default='auto',  help="""Using svd_solver randomized will increase accuracy up to 0.1 but for first run using auto is recomended""")
    parser.add_argument("--float16", default=False, type=utils.bool_flag, help="""help to using floating point 16 on your results, 
                        basic extract features from Dino-ViT is floating point 32""")
    args = parser.parse_args()
    main(args.dataset, args.act_pca, args.n_component, args.load_features, args.svd_solver, args.float16)

    #Example:
    #python3 pca_dino.py --dataset cifar10 --load_features output/ ==> without PCA
    #python3 pca_dino.py --dataset cifar10 --load_features output/ --act_pca True --n_component 20 --svd_solver randomized --float16 True==> with PCA


    # note for SVM_classifier #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    # C=1.0,            # Regularization parameter. Higher values mean stricter margin (low bias, high variance).
    # kernel='rbf',     # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid', or custom.
    # degree=3,         # Degree for 'poly' kernel. Ignored by other kernels.
    # gamma='scale',    # Kernel coefficient: 'scale', 'auto', or a float value for poly kernel.
    # coef0=0.0,        # Independent term in 'poly' and 'sigmoid' kernels.
    # tol=1e-3,         # Tolerance for stopping criterion.
    # class_weight='balanced', # Adjusts weights inversely proportional to class frequencies.
    # max_iter=-1,      # Limit on iterations within solver, -1 for no limit.
    # probability=True, # Enable probability estimates. Slower but useful.
    # random_state=42   # Seed for reproducible output.
