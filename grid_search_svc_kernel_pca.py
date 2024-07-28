import os
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import torch
from tqdm import tqdm
import joblib

# Create a pipeline with standard scaler, kernel PCA, and SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kpca', KernelPCA()),
    ('svm', SVC())
])

# Define the parameter grid
param_grid = {
    'kpca__kernel': ['rbf', 'poly', 'sigmoid'],
    'kpca__gamma': [1e-3, 1e-2, 1e-1, 1],  # For RBF and Polynomial kernels
    'kpca__degree': [2, 3, 4],  # For Polynomial kernel
    'kpca__coef0': [0, 1],  # For Polynomial and Sigmoid kernels
    'kpca__n_components': [5, 10, 20, 30, 40, 45, 50, 60, 70, 80, 120, 160, 200, 230, 270, 310, 350],  # Number of components to keep
    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # SVM kernels to explore
    'svm__C': [0.01, 0.1, 1, 10, 100],  # SVM regularization parameter
    'svm__gamma': ['scale', 'auto']  # SVM gamma parameter for RBF kernel
}

# Set up Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5)

name_dataset = "caltech101"
X_train = torch.load(os.path.join(f'output/{name_dataset}','trainfeat.pth')).cpu().numpy()
y_train = torch.load(os.path.join(f'output/{name_dataset}','trainlabels.pth')).numpy()

# Integrate tqdm with joblib
with joblib.parallel_backend('loky', n_jobs=-1):
    with tqdm(total=len(param_grid['kpca__kernel']) *
                    len(param_grid['kpca__gamma']) *
                    len(param_grid['kpca__degree']) *
                    len(param_grid['kpca__coef0']) *
                    len(param_grid['kpca__n_components']) *
                    len(param_grid['svm__kernel']) *
                    len(param_grid['svm__C']) *
                    len(param_grid['svm__gamma'])) as progress_bar:
        def tqdm_grid_search(*args, **kwargs):
            result = original_grid_search(*args, **kwargs)
            progress_bar.update(1)
            return result

        # Backup the original fit function
        original_grid_search = GridSearchCV.fit
        # Replace the fit function with the tqdm-wrapped version
        GridSearchCV.fit = tqdm_grid_search

        # Set up the GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')

        # Fit the GridSearch to your data
        grid_search.fit(X_train, y_train)

        # Restore the original fit function
        GridSearchCV.fit = original_grid_search

# Output the best parameters and the best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# Save the GridSearchCV results to a CSV file
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('grid_search_results_kpca.csv', index=False)
