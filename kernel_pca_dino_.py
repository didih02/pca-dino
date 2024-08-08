#run dino.py first, after that you will get train and test in .pth file.

import argparse
import os
import time
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import utils

def get_file_size_in_kb(file_path):
    file_size_bytes = os.path.getsize(file_path)  # Get file size in bytes
    file_size_kb = file_size_bytes / 1024  # Convert bytes to kilobytes
    return file_size_kb

def calculate_topk_accuracy(classifier, output, target, topk=(1, 5)):
    # https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        res = []        
        check_class = torch.unique(target).size(0)

        if check_class > 2:
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        else:
            print(f"Only two classes, not need top1 and top5 accuracy using {classifier}")
            res = torch.tensor([[0],[0]], dtype=torch.float)
        return res

def knn_classify(name_dataset, X_train, y_train, X_test, y_test, dino_dir, _size, act_pca, n_component, svd_solver):
    
    # Combine x_train and x_test if you want to use all data for cross-validation
    X_ = np.concatenate((X_train, X_test), axis=0)
    y_ = np.concatenate((y_train, y_test), axis=0)
    
    # Define k-fold cross-validation
    k_fold = 5
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    # Define lists to store results
    accuracies = []
    top1_accuracies = []
    top5_accuracies = []
    classification_reports = []

    # Train and evaluate KNN model with k-fold cross-validation
    for train_index, val_index in kf.split(X_, y_):
        X_train_fold = X_[train_index]
        y_train_fold = y_[train_index]
        X_val_fold = X_[val_index]
        y_val_fold = y_[val_index]

        # Train KNN model
        knn = KNeighborsClassifier(n_neighbors=20)
        knn.fit(X_train_fold, y_train_fold.ravel())

        # Predict probabilities
        y_pred_prob = knn.predict_proba(X_val_fold)
        y_pred = knn.predict(X_val_fold)

        # Calculate normal accuracy
        accuracy = (accuracy_score(y_val_fold, y_pred))*100

        # Calculate top-1 and top-5 accuracy
        y_val_tensor = torch.tensor(LabelEncoder().fit_transform(y_val_fold))
        y_pred_prob_tensor = torch.tensor(y_pred_prob)
    
        top1_acc, top5_acc = calculate_topk_accuracy("KNN", y_pred_prob_tensor, y_val_tensor)

        # Store results
        accuracies.append(accuracy)
        top1_accuracies.append(top1_acc.item())
        top5_accuracies.append(top5_acc.item())
        classification_reports.append(classification_report(y_val_fold, y_pred))

    # Calculate mean and standard deviation of results
    mean_accuracy = np.mean(accuracies)
    mean_top1_accuracy = np.mean(top1_accuracies)
    mean_top5_accuracy = np.mean(top5_accuracies)

    # Save classification report
    with open(os.path.join(dino_dir, "classification_report_knn.txt"), 'a') as fd:
        fd.write(f'Size: {_size}\n')
        fd.write(f'Mean Accuracy: {mean_accuracy:.2f}% \n')
        fd.write(f'Mean Top-1 Accuracy: {mean_top1_accuracy:.2f}% \n')
        fd.write(f'Mean Top-5 Accuracy: {mean_top5_accuracy:.2f}% \n')
        fd.write(f'Classification reports: \n')
        for report in classification_reports:
            fd.write(report + '\n')
        fd.write(f'Parameters: \n{knn.get_params()}\n\n\n')

    # Combine all results in one csv file
    if act_pca:
        with open(os.path.join(f'classify_pca_dino_svd_solver_{svd_solver}', f"report_pca_dino_knn_{svd_solver}.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{mean_accuracy:.2f}%;{mean_top1_accuracy:.2f}%;{mean_top5_accuracy:.2f}%;{n_component};{_size};{time.time()}\n')   
    else:
        with open(os.path.join('classify_dino', "report_dino_knn.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{mean_accuracy:.2f}%;{mean_top1_accuracy:.2f}%;{mean_top5_accuracy:.2f}%;{_size};{time.time()}\n')

def svm_classify(name_dataset, X_train, y_train, X_test, y_test, dino_dir, _size, act_pca, n_component, svd_solver):
    
    # Combine x_train and x_test if you want to use all data for cross-validation
    X_ = np.concatenate((X_train, X_test), axis=0)
    y_ = np.concatenate((y_train, y_test), axis=0)

    k_fold = 5
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    # Define lists to store results
    accuracies = []
    top1_accuracies = []
    top5_accuracies = []
    classification_reports = []

    # Train and evaluate SVM model with k-fold cross-validation
    for train_index, test_index in kf.split(X_, y_):
        X_train, X_test = X_[train_index], X_[test_index]
        y_train, y_test = y_[train_index], y_[test_index]

        # Train SVM model
        clf = svm.SVC(kernel="linear", verbose=False)

         #best configuration
        # clf = svm.SVC(kernel="rbf", verbose=False, C=10, gamma="scale", tol=0.001) #first best setting for cifar10 and eurosat
        # clf = svm.SVC(kernel="linear", verbose=False, C=0.01, gamma="scale", tol=0.001) #second best setting for caltech101
        # clf = svm.SVC(kernel="rbf", verbose=False, C=10, gamma="auto", tol=0.001) #third best setting for cifar100
        # clf = svm.SVC(kernel="linear", verbose=False, C=100, gamma="scale", tol=0.001) #fourth best setting for caltech256
        
        clf.fit(X_train, y_train.ravel())

        # Predict
        y_pred = clf.predict(X_test)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred) * 100

        # Calculate top-1 and top-5 accuracy
        y_pred_prob = clf.decision_function(X_test)  # Get decision function scores
        y_val_tensor = torch.tensor(LabelEncoder().fit_transform(y_test))
        y_pred_prob_tensor = torch.tensor(y_pred_prob)

        top1_acc, top5_acc = calculate_topk_accuracy("SVM", y_pred_prob_tensor, y_val_tensor)

        # Store results
        accuracies.append(accuracy)
        top1_accuracies.append(top1_acc.item())
        top5_accuracies.append(top5_acc.item())
        classification_reports.append(classification_report(y_test, y_pred))

    # Calculate mean and standard deviation of results
    mean_accuracy = np.mean(accuracies)
    mean_top1_accuracy = np.mean(top1_accuracies)
    mean_top5_accuracy = np.mean(top5_accuracies)

    # Save classification report
    with open(os.path.join(dino_dir, "classification_report_svm.txt"), 'a') as fd:
        fd.write(f'Size: {_size}\n')
        fd.write(f'Mean Accuracy: {mean_accuracy:.2f}% \n')
        fd.write(f'Mean Top-1 Accuracy: {mean_top1_accuracy:.2f}% \n')
        fd.write(f'Mean Top-5 Accuracy: {mean_top5_accuracy:.2f}% \n')
        fd.write(f'Classification reports: \n')
        for report in classification_reports:
            fd.write(report + '\n')
        fd.write(f'Parameters: \n{clf.get_params()}\n\n\n')

    # Combine all results in one csv file
    if act_pca:
        with open(os.path.join(f'classify_pca_dino_svd_solver_{svd_solver}', f"report_pca_dino_svm_{svd_solver}.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{mean_accuracy:.2f}%;{mean_top1_accuracy:.2f}%;{mean_top5_accuracy:.2f}%;{n_component};{_size};{time.time()}\n')   
    else:
        with open(os.path.join('classify_dino', "report_dino_svm.csv"), 'a') as fd:
            fd.write(f'{name_dataset};{mean_accuracy:.2f}%;{mean_top1_accuracy:.2f}%;{mean_top5_accuracy:.2f}%;{_size};{time.time()}\n')

def main(dataset, kernel_act_pca, n_component, pth, fp16):

    # Paths to data and results
    name_dataset=dataset
    # src_dir = os.path.join("images/", f"{name_dataset}")
    file_pth = os.path.join(f"{pth}", f"{name_dataset}")

    X_train = torch.load(os.path.join(file_pth,'trainfeat.pth')).cpu().numpy()
    X_test = torch.load(os.path.join(file_pth,'testfeat.pth')).cpu().numpy()
    y_train = torch.load(os.path.join(file_pth,'trainlabels.pth')).numpy()
    y_test = torch.load(os.path.join(file_pth,'testlabels.pth')).numpy()

    _size = get_file_size_in_kb(os.path.join(file_pth,'trainfeat.pth'))+get_file_size_in_kb(os.path.join(file_pth,'testfeat.pth'))

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if fp16:    
        print("features use fp16")
        X_train = X_train.astype(np.float16)
        X_test = X_test.astype(np.float16)
    else:
        print("features use fp32")
        
    if kernel_act_pca:
        print(f"kernel PCA-Dino Classifier with n_components = {n_component}")
        
        kernel_pca_dino_dir = os.path.join("classify_kernel_pca_dino", f"{name_dataset}/{n_component}")
        if not os.path.exists(kernel_pca_dino_dir):
            os.makedirs(kernel_pca_dino_dir)
        pickle.dump(sc, open(os.path.join(kernel_pca_dino_dir, 'standard_scaler.sav'), 'wb'))
        
        # PCA
        if n_component is not None:
            if n_component < 1:
                kernel_pca = KernelPCA(n_components=n_component)
                # kernel_pca = KernelPCA(n_components=n_component, kernel='rbf', eigen_solver = 'randomized')
            else:
                kernel_pca = KernelPCA(min(n_component, X_train.shape[0], X_train.shape[1]))
                # kernel_pca = KernelPCA(min(n_component, X_train.shape[0], X_train.shape[1]),  kernel='rbf', eigen_solver = 'randomized')
        else:
            kernel_pca = KernelPCA()
            # kernel_pca = KernelPCA(kernel='rbf', eigen_solver = 'randomized')
        print("Kernel-PCA start")

        X_train = kernel_pca.fit_transform(X_train)
        X_test = kernel_pca.transform(X_test) 

        if fp16:    
            print(".")
            X_train = X_train.astype(np.float16)
            X_test = X_test.astype(np.float16)
        else:
            print(".")

        pickle.dump(kernel_pca, open(os.path.join(kernel_pca_dino_dir, 'pca_model.sav'), 'wb'))

        print("Kernel-PCA Model save")

        with open(os.path.join(kernel_pca_dino_dir, 'X_train_pca-dino.npy'),'wb') as npy_file:
            np.save(npy_file, X_train)
        
        with open(os.path.join(kernel_pca_dino_dir, 'X_test_pca-dino.npy'),'wb') as npy_file:
            np.save(npy_file, X_test)

        with open(os.path.join(kernel_pca_dino_dir, 'y_train_pca-dino.npy'),'wb') as npy_file:
            np.save(npy_file, y_train)
        
        with open(os.path.join(kernel_pca_dino_dir, 'y_test_pca-dino.npy'),'wb') as npy_file:
            np.save(npy_file, y_test)

        _size_pca = get_file_size_in_kb(os.path.join(kernel_pca_dino_dir, 'X_train_pca-dino.npy'))+get_file_size_in_kb(os.path.join(kernel_pca_dino_dir, 'X_test_pca-dino.npy'))
        
        with open(os.path.join(kernel_pca_dino_dir, "kernel_pca_report.txt"),'a') as fd:
            fd.write(f'Size (kb): {_size_pca}\n')
            fd.write(f'Number of components: {kernel_pca.n_components}\n')
            fd.write(f'Number of features: {kernel_pca.n_features_in_}\n')
            fd.write(f'Parameters: {kernel_pca.get_params()}\n\n\n')
        
        svm_classify(name_dataset, X_train, y_train, X_test, y_test, kernel_pca_dino_dir, _size_pca, kernel_act_pca, n_component)
        knn_classify(name_dataset, X_train, y_train, X_test, y_test, kernel_pca_dino_dir, _size_pca, kernel_act_pca, n_component)

        print("Classify done")

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

        svm_classify(name_dataset, X_train, y_train, X_test, y_test, dino_dir, _size, kernel_act_pca, n_component)
        knn_classify(name_dataset, X_train, y_train, X_test, y_test, dino_dir, _size, kernel_act_pca, n_component)
        print("Classify done")
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser('PCA-Dino')
    parser.add_argument("--dataset", default="caltech101", type=str, help="""set your actual name of your dataset""")
    parser.add_argument("--act_pca", default=False, type=utils.bool_flag, help="""set True if you want using PCA""")
    parser.add_argument("--n_component", default=20, type=int, help="""using this if you used PCA""")
    parser.add_argument("--load_features", default=None, help="""using this for load you .pth, npy, or pt file to train and test,
                        there are four file which you need first trainfeat, testfeat, trainlabels, and testlabels""")
    parser.add_argument("--float16", default=False, type=utils.bool_flag, help="""help to using floating point 16 on your results, 
                        basic extract features from Dino-ViT is floating point 32""")
    args = parser.parse_args()
    main(args.dataset, args.act_pca, args.n_component, args.load_features, args.float16)

    #Example:
    #python3 kernel_pca_dino_.py --dataset cifar10 --load_features output/ ==> without PCA
    #python3 kernel_pca_dino_.py --dataset cifar_10 --load_features output/ --act_pca True --n_component 20 --svd_solver randomized --float16 True==> with PCA


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