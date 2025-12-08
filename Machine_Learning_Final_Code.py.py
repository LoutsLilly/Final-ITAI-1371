#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load your dataset
df = pd.read_csv("heart.csv")

print("Raw data preview:")
display(df.head())
print("Shape:", df.shape)


# In[4]:


# ---- 1.1 Choose your targets ----
# TODO: adjust these if your column names are different

target_clf = 'target'  
target_reg = 'chol'     
# Keep a copy of original
df_original = df.copy()

# ---- 1.2 Feature Engineering on full df ----

# Age group bins
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 35, 45, 55, 65, 100],
    labels=['<35', '35-44', '45-54', '55-64', '65+']
)

# Cholesterol per age
if 'chol' in df.columns:
    df['chol_per_age'] = df['chol'] / df['age']

# Blood pressure / cholesterol ratio
if 'trestbps' in df.columns and 'chol' in df.columns:
    df['bp_chol_ratio'] = df['trestbps'] / df['chol']

print("After feature engineering:")
display(df.head())


# In[5]:


# Separate targets
y_clf = df[target_clf]      # classification target
y_reg = df[target_reg]      # regression target

# Drop targets from features
X = df.drop(columns=[target_clf, target_reg])

print("Feature columns:")
print(X.columns.tolist())


# In[6]:


# ---- Missing values ----
for col in X.columns:
    if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
        X[col] = X[col].fillna(X[col].mode()[0])
    else:
        X[col] = X[col].fillna(X[col].mean())

# ---- Encode categoricals ----
cat_cols = X.select_dtypes(include=['object', 'category']).columns

# Binary categoricals → LabelEncoder
for col in cat_cols:
    if X[col].nunique() <= 2:
        X[col] = LabelEncoder().fit_transform(X[col])

# Remaining categoricals → One-hot
cat_cols = X.select_dtypes(include=['object', 'category']).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print("After encoding:")
display(X.head())
print("New shape:", X.shape)


# In[7]:


# ---- Missing values ----
for col in X.columns:
    if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
        X[col] = X[col].fillna(X[col].mode()[0])
    else:
        X[col] = X[col].fillna(X[col].mean())

# ---- Encode categoricals ----
cat_cols = X.select_dtypes(include=['object', 'category']).columns

# Binary categoricals → LabelEncoder
for col in cat_cols:
    if X[col].nunique() <= 2:
        X[col] = LabelEncoder().fit_transform(X[col])

# Remaining categoricals → One-hot
cat_cols = X.select_dtypes(include=['object', 'category']).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print("After encoding:")
display(X.head())
print("New shape:", X.shape)


# In[10]:


# Targets - classification
y_clf_train = y_clf.iloc[idx_train]
y_clf_val   = y_clf.iloc[idx_val]
y_clf_test  = y_clf.iloc[idx_test]

# Targets - regression
y_reg_train = y_reg.iloc[idx_train]
y_reg_val   = y_reg.iloc[idx_val]
y_reg_test  = y_reg.iloc[idx_test]

print("Shapes:")
print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)


# In[12]:


scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)


# In[13]:


reg_models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

reg_results = []

for name, model in reg_models.items():
    model.fit(X_train_s, y_reg_train)

    y_val_pred = model.predict(X_val_s)
    y_test_pred = model.predict(X_test_s)

    reg_results.append({
        "Model": name,
        "Val_MAE": mean_absolute_error(y_reg_val, y_val_pred),
        "Val_MSE": mean_squared_error(y_reg_val, y_val_pred),
        "Val_R2":  r2_score(y_reg_val, y_val_pred),
        "Test_MAE": mean_absolute_error(y_reg_test, y_test_pred),
        "Test_MSE": mean_squared_error(y_reg_test, y_test_pred),
        "Test_R2":  r2_score(y_reg_test, y_test_pred),
    })

reg_results_df = pd.DataFrame(reg_results)
print("Regression model comparison:")
display(reg_results_df.sort_values("Val_MSE"))


# In[14]:


# Pick top 3 by lowest Val_MSE
top3_reg = reg_results_df.sort_values("Val_MSE").head(3)["Model"].tolist()
print("Top 3 regression models:", top3_reg)

voting_reg = VotingRegressor(
    estimators=[(name, reg_models[name]) for name in top3_reg]
)

voting_reg.fit(X_train_s, y_reg_train)

y_val_v = voting_reg.predict(X_val_s)
y_test_v = voting_reg.predict(X_test_s)

voting_reg_metrics = {
    "Model": "VotingRegressor",
    "Val_MAE": mean_absolute_error(y_reg_val, y_val_v),
    "Val_MSE": mean_squared_error(y_reg_val, y_val_v),
    "Val_R2":  r2_score(y_reg_val, y_val_v),
    "Test_MAE": mean_absolute_error(y_reg_test, y_test_v),
    "Test_MSE": mean_squared_error(y_reg_test, y_test_v),
    "Test_R2":  r2_score(y_reg_test, y_test_v),
}

reg_results_df = pd.concat(
    [reg_results_df, pd.DataFrame([voting_reg_metrics])],
    ignore_index=True
)

print("With VotingRegressor added:")
display(reg_results_df.sort_values("Val_MSE"))


# In[15]:


# Use same top 3 models for Bayesian averaging
val_mse_top3 = reg_results_df.set_index("Model").loc[top3_reg, "Val_MSE"]

# Convert MSE to weights (smaller MSE => larger weight)
inv_errors = np.exp(-val_mse_top3)
weights_reg = inv_errors / inv_errors.sum()
print("Bayesian regression weights:")
display(weights_reg)

# Collect predictions
val_preds = np.array([reg_models[name].predict(X_val_s) for name in top3_reg])
test_preds = np.array([reg_models[name].predict(X_test_s) for name in top3_reg])

w = weights_reg.values.reshape(-1, 1)

y_val_bayes_reg = (w * val_preds).sum(axis=0)
y_test_bayes_reg = (w * test_preds).sum(axis=0)

bayesian_reg_metrics = {
    "Model": "BayesianEnsemble_Reg",
    "Val_MAE": mean_absolute_error(y_reg_val, y_val_bayes_reg),
    "Val_MSE": mean_squared_error(y_reg_val, y_val_bayes_reg),
    "Val_R2":  r2_score(y_reg_val, y_val_bayes_reg),
    "Test_MAE": mean_absolute_error(y_reg_test, y_test_bayes_reg),
    "Test_MSE": mean_squared_error(y_reg_test, y_test_bayes_reg),
    "Test_R2":  r2_score(y_reg_test, y_test_bayes_reg),
}

reg_results_df = pd.concat(
    [reg_results_df, pd.DataFrame([bayesian_reg_metrics])],
    ignore_index=True
)

print("Final regression results (including ensembles):")
display(reg_results_df.sort_values("Val_MSE"))


# In[16]:


binary = (len(np.unique(y_clf)) == 2)

clf_models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVC": SVC(probability=True, random_state=42)
}

clf_results = []

for name, model in clf_models.items():
    model.fit(X_train_s, y_clf_train)

    y_val_pred = model.predict(X_val_s)
    y_test_pred = model.predict(X_test_s)

    y_val_prob = model.predict_proba(X_val_s)[:, 1] if binary else None
    y_test_prob = model.predict_proba(X_test_s)[:, 1] if binary else None

    val_acc  = accuracy_score(y_clf_val, y_val_pred)
    val_prec = precision_score(y_clf_val, y_val_pred, average="binary" if binary else "macro")
    val_rec  = recall_score(y_clf_val, y_val_pred, average="binary" if binary else "macro")
    val_f1   = f1_score(y_clf_val, y_val_pred, average="binary" if binary else "macro")
    val_auc  = roc_auc_score(y_clf_val, y_val_prob) if binary else np.nan

    test_acc  = accuracy_score(y_clf_test, y_test_pred)
    test_prec = precision_score(y_clf_test, y_test_pred, average="binary" if binary else "macro")
    test_rec  = recall_score(y_clf_test, y_test_pred, average="binary" if binary else "macro")
    test_f1   = f1_score(y_clf_test, y_test_pred, average="binary" if binary else "macro")
    test_auc  = roc_auc_score(y_clf_test, y_test_prob) if binary else np.nan

    clf_results.append({
        "Model": name,
        "Val_Accuracy": val_acc,
        "Val_Precision": val_prec,
        "Val_Recall": val_rec,
        "Val_F1": val_f1,
        "Val_ROC_AUC": val_auc,
        "Test_Accuracy": test_acc,
        "Test_Precision": test_prec,
        "Test_Recall": test_rec,
        "Test_F1": test_f1,
        "Test_ROC_AUC": test_auc
    })

clf_results_df = pd.DataFrame(clf_results)
print("Classification model comparison:")
display(clf_results_df.sort_values("Val_F1", ascending=False))


# In[17]:


top_metric = "Val_F1"  # you could also choose "Val_ROC_AUC" if binary

top3_clf = clf_results_df.sort_values(top_metric, ascending=False).head(3)["Model"].tolist()
print("Top 3 classification models:", top3_clf)

voting_clf = VotingClassifier(
    estimators=[(name, clf_models[name]) for name in top3_clf],
    voting="soft"
)

voting_clf.fit(X_train_s, y_clf_train)

y_val_pred_v = voting_clf.predict(X_val_s)
y_test_pred_v = voting_clf.predict(X_test_s)

y_val_prob_v = voting_clf.predict_proba(X_val_s)[:, 1] if binary else None
y_test_prob_v = voting_clf.predict_proba(X_test_s)[:, 1] if binary else None

voting_clf_metrics = {
    "Model": "VotingClassifier",
    "Val_Accuracy": accuracy_score(y_clf_val, y_val_pred_v),
    "Val_Precision": precision_score(y_clf_val, y_val_pred_v, average="binary" if binary else "macro"),
    "Val_Recall": recall_score(y_clf_val, y_val_pred_v, average="binary" if binary else "macro"),
    "Val_F1": f1_score(y_clf_val, y_val_pred_v, average="binary" if binary else "macro"),
    "Val_ROC_AUC": roc_auc_score(y_clf_val, y_val_prob_v) if binary else np.nan,
    "Test_Accuracy": accuracy_score(y_clf_test, y_test_pred_v),
    "Test_Precision": precision_score(y_clf_test, y_test_pred_v, average="binary" if binary else "macro"),
    "Test_Recall": recall_score(y_clf_test, y_test_pred_v, average="binary" if binary else "macro"),
    "Test_F1": f1_score(y_clf_test, y_test_pred_v, average="binary" if binary else "macro"),
    "Test_ROC_AUC": roc_auc_score(y_clf_test, y_test_prob_v) if binary else np.nan
}

clf_results_df = pd.concat(
    [clf_results_df, pd.DataFrame([voting_clf_metrics])],
    ignore_index=True
)

print("With VotingClassifier added:")
display(clf_results_df.sort_values("Val_F1", ascending=False))


# In[18]:


# Use F1 scores as performance measure for weights
val_f1_top3 = clf_results_df.set_index("Model").loc[top3_clf, "Val_F1"]

scores = np.exp(val_f1_top3)
weights_clf = scores / scores.sum()
print("Bayesian classification weights:")
display(weights_clf)

# Collect probabilities
val_probs = np.array([
    clf_models[name].predict_proba(X_val_s)[:, 1]
    for name in top3_clf
])
test_probs = np.array([
    clf_models[name].predict_proba(X_test_s)[:, 1]
    for name in top3_clf
])

w = weights_clf.values.reshape(-1, 1)

y_val_prob_bayes = (w * val_probs).sum(axis=0)
y_test_prob_bayes = (w * test_probs).sum(axis=0)

y_val_pred_bayes = (y_val_prob_bayes >= 0.5).astype(int)
y_test_pred_bayes = (y_test_prob_bayes >= 0.5).astype(int)

bayes_clf_metrics = {
    "Model": "BayesianEnsemble_Clf",
    "Val_Accuracy": accuracy_score(y_clf_val, y_val_pred_bayes),
    "Val_Precision": precision_score(y_clf_val, y_val_pred_bayes, average="binary"),
    "Val_Recall": recall_score(y_clf_val, y_val_pred_bayes, average="binary"),
    "Val_F1": f1_score(y_clf_val, y_val_pred_bayes, average="binary"),
    "Val_ROC_AUC": roc_auc_score(y_clf_val, y_val_prob_bayes),
    "Test_Accuracy": accuracy_score(y_clf_test, y_test_pred_bayes),
    "Test_Precision": precision_score(y_clf_test, y_test_pred_bayes, average="binary"),
    "Test_Recall": recall_score(y_clf_test, y_test_pred_bayes, average="binary"),
    "Test_F1": f1_score(y_clf_test, y_test_pred_bayes, average="binary"),
    "Test_ROC_AUC": roc_auc_score(y_clf_test, y_test_prob_bayes)
}

clf_results_df = pd.concat(
    [clf_results_df, pd.DataFrame([bayes_clf_metrics])],
    ignore_index=True
)

print("Final classification results (including ensembles):")
display(clf_results_df.sort_values("Val_F1", ascending=False))


# In[19]:


# Use F1 scores as performance measure for weights
val_f1_top3 = clf_results_df.set_index("Model").loc[top3_clf, "Val_F1"]

scores = np.exp(val_f1_top3)
weights_clf = scores / scores.sum()
print("Bayesian classification weights:")
display(weights_clf)

# Collect probabilities
val_probs = np.array([
    clf_models[name].predict_proba(X_val_s)[:, 1]
    for name in top3_clf
])
test_probs = np.array([
    clf_models[name].predict_proba(X_test_s)[:, 1]
    for name in top3_clf
])

w = weights_clf.values.reshape(-1, 1)

y_val_prob_bayes = (w * val_probs).sum(axis=0)
y_test_prob_bayes = (w * test_probs).sum(axis=0)

y_val_pred_bayes = (y_val_prob_bayes >= 0.5).astype(int)
y_test_pred_bayes = (y_test_prob_bayes >= 0.5).astype(int)

bayes_clf_metrics = {
    "Model": "BayesianEnsemble_Clf",
    "Val_Accuracy": accuracy_score(y_clf_val, y_val_pred_bayes),
    "Val_Precision": precision_score(y_clf_val, y_val_pred_bayes, average="binary"),
    "Val_Recall": recall_score(y_clf_val, y_val_pred_bayes, average="binary"),
    "Val_F1": f1_score(y_clf_val, y_val_pred_bayes, average="binary"),
    "Val_ROC_AUC": roc_auc_score(y_clf_val, y_val_prob_bayes),
    "Test_Accuracy": accuracy_score(y_clf_test, y_test_pred_bayes),
    "Test_Precision": precision_score(y_clf_test, y_test_pred_bayes, average="binary"),
    "Test_Recall": recall_score(y_clf_test, y_test_pred_bayes, average="binary"),
    "Test_F1": f1_score(y_clf_test, y_test_pred_bayes, average="binary"),
    "Test_ROC_AUC": roc_auc_score(y_clf_test, y_test_prob_bayes)
}

clf_results_df = pd.concat(
    [clf_results_df, pd.DataFrame([bayes_clf_metrics])],
    ignore_index=True
)

print("Final classification results (including ensembles):")
display(clf_results_df.sort_values("Val_F1", ascending=False))

