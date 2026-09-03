"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    for i in range(X.shape[1]):
        mean=0
        ct=0
        for j in range(X.shape[0]):
            if np.isnan(X[j][i]):
                continue
            mean+=X[j][i]
            ct+=1
        if(ct!=0):
            mean/=ct
        for j in range(X.shape[0]):
            if np.isnan(X[j][i]):
                X[j][i]=mean
    return X

# Step 2 - compute_iqr_bounds
import numpy as np
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    quantile1=np.percentile(X,25,axis=0)
    quantile2=np.percentile(X,75,axis=0)
    low=[]
    up=[]
    for i in range(X.shape[1]):
        iqr=quantile2[i]-quantile1[i]
        lower=quantile1[i]-k*iqr
        upper=quantile2[i]+k*iqr
        low.append(lower)
        up.append(upper)
    lower=np.array(low)
    upper=np.array(up)
    return (lower,upper)

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    X_Copy=X.copy()
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            if X_Copy[j][i]>upper[i]:
                X_Copy[j][i]=upper[i]
            if X_Copy[j][i]<lower[i]:
                X_Copy[j][i]=lower[i]
    return X_Copy

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    ratio=[]
    for i in range(numerator.shape[0]):
        ratio.append(numerator[i]/(denominator[i]+eps))
    return np.array(ratio)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X_new=np.column_stack((X,col))
    return X_new

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    unique=sorted(set(labels))
    encoded=[]
    mapped={}
    for i in range(len(unique)):
        mapped[unique[i]]=i
    for i in range(len(labels)):
        temp=[0.0]*len(unique)
        temp[mapped[labels[i]]]=1.0
        encoded.append(temp)
    return np.array(encoded)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0,ddof=0)
    for i in range(len(std)):
        if(std[i]==0.0):
            std[i]=1.0
    return (mean,std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X-mean)/std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    n=X.shape[0]
    bias=np.ones(n,dtype=float)
    X_new=np.insert(X,0,bias,axis=1)
    return X_new

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng=np.random.default_rng(seed)
    indices=rng.permutation(n_samples)
    return indices

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    N=len(indices)
    train_ind=indices[0:int(N*train_ratio)]
    val_ind=indices[int(N*train_ratio):int(N*train_ratio)+int(N*val_ratio)]
    test_ind=indices[int(N*val_ratio)+int(N*train_ratio):N]
    return (train_ind,val_ind,test_ind)

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    n=X.shape[0]
    X_sub=X[indices]
    y_sub=y[indices]
    return X_sub,y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    A=X.T @ X
    # if(np.linalg.det(A)!=0):
    #     A_inv=np.linalg.inv(A)
    B=X.T @ y
    return np.linalg.solve(A,B)

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    predict=X @ theta
    return predict

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    mae=0.0
    n=len(y_true)
    for i in range(n):
        mae+=abs(y_true[i]-y_pred[i])
    return mae/n

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    n=len(y_pred)
    rmse=0.0
    for i in range(n):
        rmse+=(y_pred[i]-y_true[i])**2
    rmse/=n
    rmse=rmse**0.5
    return rmse

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    sse=np.sum(np.square(y_true-y_pred))
    sst=np.sum(np.square(y_true-np.mean(y_true)))
    if sst==0 :
        return 0.0
    return 1-sse/sst

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r=y_true-y_pred
    return {"mean":float(np.mean(r)),"std":float(np.std(r)),"median_abs":float(np.median(np.abs(r)))}

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X=impute_nan_with_mean(X)
    lower,upper=compute_iqr_bounds(X,iqr_k)
    X_clipped=clip_columns(X,lower,upper)
    return X_clipped

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    x_num=X_num[:,ratio_num_idx]
    x_den=X_num[:,ratio_den_idx]
    ratio=make_ratio_feature(x_num,x_den)
    X_num=append_column(X_num,ratio)
    if cat_labels is None:
        return X_num
    else:
        unique=sorted(set(cat_labels))
        encoded=[]
        mapped={}
        for i in range(len(unique)):
            mapped[unique[i]]=i
        for i in range(len(cat_labels)):
            temp=[0.0]*len(unique)
            temp[mapped[cat_labels[i]]]=1.0
            encoded.append(temp)
        Cat_labels=np.array(encoded)
        X_num=append_column(X_num,Cat_labels)
        return X_num

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    np.random.seed(seed)
    indices=np.random.permutation(X.shape[0])
    N=X.shape[0]
    train_ind=indices[0:int(N*train_ratio)]
    val_ind=indices[int(N*train_ratio):int(N*train_ratio)+int(N*val_ratio)]
    test_ind=indices[int(N*val_ratio)+int(N*train_ratio):N]
    x_train=X[train_ind]
    y_train=y[train_ind]
    X_val=X[val_ind]
    y_val=y[val_ind]
    x_test=X[test_ind]
    y_test=y[test_ind]
    return {"X_train":x_train,"y_train":y_train,"X_val":X_val,"y_val":y_val,"X_test":x_test,"y_test":y_test}

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    mean=np.mean(splits['X_train'],axis=0)
    std=np.std(splits['X_train'],axis=0)
    for i in range(len(std)):
        if std[i]==0.0:
            std[i]=1.0
    x_train_std=(splits['X_train']-mean)/std
    x_val_std=(splits['X_val']-mean)/std
    x_test_std=(splits['X_test']-mean)/std
    bias=np.ones(splits['X_train'].shape[0],dtype=float)
    bias1=np.ones(splits['X_val'].shape[0],dtype=float)
    bias2=np.ones(splits['X_test'].shape[0],dtype=float)

    x_train_std=np.insert(x_train_std,0,bias,axis=1)
    x_val_std=np.insert(x_val_std,0,bias1,axis=1)
    x_test_std=np.insert(x_test_std,0,bias2,axis=1)
    std_splits={"X_train":x_train_std,"X_test":x_test_std,"X_val":x_val_std,"y_train":splits["y_train"],"y_test":splits["y_test"],"y_val":splits["y_val"]}
    return (std_splits,mean,std)

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    mae=mean_absolute_error(y_true,y_pred)
    rmse=root_mean_squared_error(y_true,y_pred)
    r2=r_squared(y_true,y_pred)
    res=residual_summary(y_true,y_pred)
    return {"mae":mae,"rmse":rmse,"r2":r2,"residual_summary":res}

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...
    X = np.asarray(X, dtype=float)

    col_mean = np.nanmean(X, axis=0)

    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])

    # ---------------------------------------------------------
    # 2. Compute IQR bounds and clip columns
    # ---------------------------------------------------------
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)

    iqr = q3 - q1

    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr

    X = np.clip(X, lower, upper)

    # ---------------------------------------------------------
    # 3. Create ratio feature
    # ---------------------------------------------------------
    x_num = X[:, ratio_num_idx]
    x_den = X[:, ratio_den_idx]

    # Same idea as make_ratio_feature()
    # Avoid division by zero
    ratio = np.divide(
        x_num,
        x_den,
        out=np.zeros_like(x_num, dtype=float),
        where=x_den != 0
    )

    # Append ratio as a new column
    X = np.column_stack((X, ratio))

    # ---------------------------------------------------------
    # 4. One-hot encode categorical labels
    # ---------------------------------------------------------
    if cat_labels is not None:

        unique = sorted(set(cat_labels))

        mapped = {}
        for i in range(len(unique)):
            mapped[unique[i]] = i

        encoded = []

        for label in cat_labels:
            temp = [0.0] * len(unique)
            temp[mapped[label]] = 1.0
            encoded.append(temp)

        cat_encoded = np.array(encoded, dtype=float)

        # Append categorical features
        X = np.column_stack((X, cat_encoded))

    # ---------------------------------------------------------
    # 5. Train / validation / test split
    # ---------------------------------------------------------
    np.random.seed(seed)

    indices = np.random.permutation(X.shape[0])
    N = X.shape[0]

    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)

    train_ind = indices[:train_end]
    val_ind = indices[train_end:val_end]
    test_ind = indices[val_end:N]

    X_train = X[train_ind]
    y_train = y[train_ind]

    X_val = X[val_ind]
    y_val = y[val_ind]

    X_test = X[test_ind]
    y_test = y[test_ind]

    # ---------------------------------------------------------
    # 6. Standardize using ONLY training statistics
    # ---------------------------------------------------------
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    for i in range(len(std)):
        if std[i] == 0.0:
            std[i] = 1.0

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std

    # ---------------------------------------------------------
    # 7. Add bias column
    # ---------------------------------------------------------
    bias_train = np.ones(X_train_std.shape[0], dtype=float)
    bias_val = np.ones(X_val_std.shape[0], dtype=float)
    bias_test = np.ones(X_test_std.shape[0], dtype=float)

    X_train_std = np.insert(X_train_std, 0, bias_train, axis=1)
    X_val_std = np.insert(X_val_std, 0, bias_val, axis=1)
    X_test_std = np.insert(X_test_std, 0, bias_test, axis=1)

    # ---------------------------------------------------------
    # 8. OLS using Normal Equation
    # ---------------------------------------------------------
    # w = (X^T X)^(-1) X^T y
    #
    # pinv is safer than directly using inv()
    w = np.linalg.pinv(X_train_std.T @ X_train_std) @ \
        X_train_std.T @ y_train

    # ---------------------------------------------------------
    # 9. Predictions
    # ---------------------------------------------------------
    y_train_pred = X_train_std @ w
    y_val_pred = X_val_std @ w
    y_test_pred = X_test_std @ w

    # ---------------------------------------------------------
    # 10. Evaluate
    # ---------------------------------------------------------
    train_metrics = evaluate_predictions(y_train, y_train_pred)
    val_metrics = evaluate_predictions(y_val, y_val_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    # ---------------------------------------------------------
    # 11. Return everything useful
    # ---------------------------------------------------------
    return {
        "theta": w,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }

