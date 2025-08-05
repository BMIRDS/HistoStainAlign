# Standard Library Imports
import os
import argparse
import numpy as np
import pandas as pd

# Third Party Imports
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, 
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_curve
)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
from numpy.random import default_rng

def plot_roc_curve(test_labels, pred_scores, model_name="model"):
    """Plots the ROC curve and calculates AUC."""

    pred_scores = np.array(pred_scores)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(test_labels, pred_scores[:, 1])  # Use positive class scores
    auc_score = roc_auc_score(test_labels, pred_scores[:, 1])

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"{model_name} (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"roc_curve_{model_name}.png")

    return auc_score

def bootstrap_confidence_intervals(y_true, y_pred, pred_scores, n_bootstraps=1000, ci=95):
    """
    Calculate confidence intervals for AUC, F1, Precision, and Recall using bootstrapping.
    """

    # fixed random seed so this function always tests on same data
    rng = default_rng(42)

    auc_scores, f1_scores, precision_scores, recall_scores = [], [], [], []
    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        pred_scores_boot = np.array(pred_scores)[indices]
        
        # Calculate metrics for the bootstrap sample
        auc = roc_auc_score(y_true_boot, pred_scores_boot[:, 1])
        
        f1 = f1_score(y_true_boot, y_pred_boot, average='weighted')
        precision = precision_score(y_true_boot, y_pred_boot, average='macro')
        recall = recall_score(y_true_boot, y_pred_boot, average='macro')

        auc_scores.append(auc)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Calculate confidence intervals
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    ci_metrics = {
        "AUC": (np.percentile(auc_scores, lower_percentile), np.percentile(auc_scores, upper_percentile)),
        "F1": (np.percentile(f1_scores, lower_percentile), np.percentile(f1_scores, upper_percentile)),
        "Precision": (np.percentile(precision_scores, lower_percentile), np.percentile(precision_scores, upper_percentile)),
        "Recall": (np.percentile(recall_scores, lower_percentile), np.percentile(recall_scores, upper_percentile))
    }

    return ci_metrics, auc_scores, f1_scores, precision_scores, recall_scores

def plot_confusion_matrix(cm, model: str, class_labels, cmap="Blues"):
    """
    Saves a confusion matrix with both absolute and percentage values.
    """

    if isinstance(cm, pd.DataFrame):
        cm = cm.values
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations as strings with absolute and percentage values
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=annot, 
        fmt='',       
        cmap=cmap,
        cbar=True,
        annot_kws={"size": 14}
    )

    plt.xlabel("Prediction", fontsize=14)
    plt.ylabel("Reference Standard", fontsize=14)
    plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=45, ha="right", fontsize=14)
    plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=0, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model}.png")

def calculate_metrics(y_true, y_pred, pred_scores):
    """
    Calculate and print various evaluation metrics.
    
    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - y_scores: Target scores (for AUC).
    """

    if len(np.unique(y_true)) > 2:
        # multi-class 
        auc = roc_auc_score(y_true, pred_scores, multi_class="ovr", average="macro",)
    else:
        # regular 
        auc = roc_auc_score(y_true, pred_scores[:, 1]) # only send positive class score)
    bacc = balanced_accuracy_score(y_true, y_pred)
    return auc, bacc

def load_embeddings_and_labels(embeddings_path, df):
    """
    Load embeddings and labels from the given directory and DataFrame.
    """

    labels, features = [], []
    for file in os.listdir(embeddings_path):
        slide_id = f"{file.split('.')[0]}.svs"
        slide_label = df[df['slide_id'] == slide_id]['label'].values[0]
        labels.append(slide_label)
        features.append(torch.load(f"{embeddings_path}/{file}"))
    return labels, features

def load_and_split(fold, model, train_df, test_df, embeddings_dir, eval_type):
    """
    Returns scaled train and val features and labels for a given fold and model
    """
    embeddings_path_train = f"{embeddings_dir}/{fold}/{model}/train"
    embeddings_path_test = f"{embeddings_dir}/{fold}/{model}/{eval_type}"

    train_labels, train_features = load_embeddings_and_labels(embeddings_path_train, train_df)
    test_labels, test_features = load_embeddings_and_labels(embeddings_path_test, test_df)

    train_features = torch.stack(train_features).squeeze(1) # Shape: [num_train_samples, embed_dim]
    test_features = torch.stack(test_features).squeeze(1)   # Shape: [num_test_samples, embed_dim]
    train_labels = torch.tensor(train_labels)               # Shape: [num_train_samples]

    pipeline = Pipeline([('scaler', StandardScaler())])
    train_features = pipeline.fit_transform(train_features)
    test_features = pipeline.transform(test_features)

    return train_features, train_labels, test_features, test_labels

def eval_linprobe(fold, model, train_df, test_df, embeddings_dir, agg_cm=None, eval_type='test'):
    """
    Runs a logistic regression on the given model and fold, and returns the aggregated confusion matrix.
    Adapted from https://github.com/mahmoodlab/TANGLE/blob/main/run_linear_probing.py
    """
    train_features, train_labels, test_features, test_labels = load_and_split(fold, model, train_df, test_df, embeddings_dir, eval_type)
    
    # grid search to find best parameters
    param_dist = {'C': np.logspace(-3, 2, 100)}
    clf = RandomizedSearchCV(
        LogisticRegression(max_iter=10000, class_weight='balanced', random_state=0),
        param_dist,
        scoring='average_precision', # average_precision
        n_iter=40,
        cv=5,
        random_state=0
    )

    clf.fit(X=train_features, y=train_labels)
    pred_labels = clf.predict(X=test_features)
    pred_scores = clf.predict_proba(X=test_features)

    # Compute confusion matrix for this fold
    cm = pd.crosstab(pd.Series(test_labels, name='True'), 
                     pd.Series(pred_labels, name='Predicted'), 
                     margins=False)

    # Initialize or update the aggregated confusion matrix
    if agg_cm is None:
        agg_cm = cm.copy()
    else:
        agg_cm = agg_cm.add(cm, fill_value=0)

    return agg_cm, test_labels, pred_labels, pred_scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate linear probe on slide embeddings')
    parser.add_argument('--study', type=str, choices=['pdl1', 'p53', 'ki67'], help='Study to evaluate linear probe on')
    parser.add_argument('--model', type=str, help='name of model checkpoint to evaluate')
    parser.add_argument('--folds', type=list, default=[0, 1, 2, 3, 4], help='List of folds to evaluate')
    parser.add_argument('--embeddings_dir_baseline', type=str, help='Directory containing embeddings')
    parser.add_argument('--embeddings_dir_new', type=str, help='Directory containing embeddings')
    args = parser.parse_args()

    # Initialize lists to store aggregated metrics across folds
    baseline_tp_scores, new_tp_scores = [], []
    baseline_tn_scores, new_tn_scores = [], []

    embeddings_dirs = [args.embeddings_dir_baseline, args.embeddings_dir_new]
    for embeddings_dir in embeddings_dirs:
        test_labels_agg, pred_labels_agg, pred_scores_agg = [], [], []
        agg_cm = None   # to be aggregated across folds for a given model
        for fold in args.folds:
            # obtaining the splits specific to the study
            if args.study == 'pdl1':
                train_df = pd.read_csv(f'../pdl1_project/prov-gigapath/dataset_csv/pdl1/train_{fold}.csv')
                val_df = pd.read_csv(f'../pdl1_project/prov-gigapath/dataset_csv/pdl1/val_{fold}.csv')
                test_df = pd.read_csv(f'../pdl1_project/prov-gigapath/dataset_csv/pdl1/test_{fold}.csv')
            elif args.study == 'p53':
                train_df = pd.read_csv(f'../prov-gigapath/dataset_csv/p53/train_{fold}.csv')
                val_df = pd.read_csv(f'../prov-gigapath/dataset_csv/p53/val_{fold}.csv')
                test_df = pd.read_csv(f'../prov-gigapath/dataset_csv/p53/test_{fold}.csv')
            elif args.study == 'ki67':
                train_df = pd.read_csv(f'../ki67_project/prov-gigapath/dataset_csv/ki67/train_{fold}.csv')
                val_df = pd.read_csv(f'../ki67_project/prov-gigapath/dataset_csv/ki67/val_{fold}.csv')
                test_df = pd.read_csv(f'../ki67_project/prov-gigapath/dataset_csv/ki67/test_{fold}.csv')

            # evaluate the linear probe
            agg_cm, test_labels, pred_labels, pred_scores = eval_linprobe(fold, args.model, train_df, test_df, embeddings_dir, agg_cm)

            # Calculate metrics for this fold
            auc_fold, bacc_fold = calculate_metrics(test_labels, pred_labels, np.array(pred_scores))
            f1_agg_fold = f1_score(test_labels, pred_labels, average='weighted')
            precision_fold = precision_score(test_labels, pred_labels, average='macro')
            recall_fold = recall_score(test_labels, pred_labels, average='macro')

            test_labels_agg.extend(test_labels)
            pred_labels_agg.extend(pred_labels)
            pred_scores_agg.extend(pred_scores)
        
        print(f"Confusion matrix for model {args.model}")
        print(agg_cm)
        # plot_confusion_matrix(agg_cm, embeddings_dir.split('/')[1], ['Low Expression', 'Intermediate/High Expression'])

        # display aggregated metrics for model
        auc, bacc = calculate_metrics(test_labels_agg, pred_labels_agg, np.array(pred_scores_agg))
        f1_agg = f1_score(test_labels_agg, pred_labels_agg, average='weighted')
        precision = precision_score(test_labels_agg, pred_labels_agg, average='macro')
        recall = recall_score(test_labels_agg, pred_labels_agg, average='macro')
        pred_scores_agg = np.array(pred_scores_agg)
        ci_metrics, _, _, _, _ = bootstrap_confidence_intervals(test_labels_agg, pred_labels_agg, pred_scores_agg)
        print(
            f"Aggregated AUC: {auc:.3f} ({ci_metrics['AUC'][0]:.3f} - {ci_metrics['AUC'][1]:.3f})\n"
            f"F1: {f1_agg:.3f} ({ci_metrics['F1'][0]:.3f} - {ci_metrics['F1'][1]:.3f})\n"
            f"Precision: {precision:.3f} ({ci_metrics['Precision'][0]:.3f} - {ci_metrics['Precision'][1]:.3f})\n"
            f"Recall: {recall:.3f} ({ci_metrics['Recall'][0]:.3f} - {ci_metrics['Recall'][1]:.3f})"
        )

        if embeddings_dir == args.embeddings_dir_baseline:
            tp_indices = np.where(np.array(test_labels_agg) == 1)
            tn_indices = np.where(np.array(test_labels_agg) == 0)
            baseline_tp_scores.extend(np.array(pred_scores_agg)[tp_indices, 1].flatten())
            baseline_tn_scores.extend(np.array(pred_scores_agg)[tn_indices, 0].flatten())
        else:
            tp_indices = np.where(np.array(test_labels_agg) == 1)
            tn_indices = np.where(np.array(test_labels_agg) == 0)
            new_tp_scores.extend(np.array(pred_scores_agg)[tp_indices, 1].flatten())
            new_tn_scores.extend(np.array(pred_scores_agg)[tn_indices, 0].flatten())

    # Significance testing to compare the two models (new against baseline)
    _, p_value = wilcoxon(baseline_tp_scores, new_tp_scores, alternative='less')
    print(f"Wilcoxon signed-rank test p-value: {p_value:.3f}")

    # Significance testing to compare the two models (new against baseline)
    _, p_value = wilcoxon(baseline_tn_scores, new_tn_scores, alternative='less')
    print(f"Wilcoxon signed-rank test p-value: {p_value:.3f}")
