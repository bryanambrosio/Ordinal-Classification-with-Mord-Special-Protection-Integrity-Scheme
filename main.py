import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mord import OrdinalRidge, LogisticIT, LogisticAT  # Ordinal regression models from mord library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Principal Component Analysis for visualization
from imblearn.over_sampling import SMOTE  # Synthetic Minority Oversampling Technique for class balancing
from collections import OrderedDict, Counter
from sklearn.metrics import classification_report
import os

# Create output directory if it doesn't exist
outdir = "Ordinal Classifier"
os.makedirs(outdir, exist_ok=True)

# Load dataset from Excel file
file_path = 'dadosfiguras1500_minUG.xlsx'
df = pd.read_excel(file_path)

# Basic data filtering to ensure quality (difference of angle threshold)
df = df[df['Vang_XES_0+'] - df['Vang_XES_0-'] >= 1].copy()

# Convert target column to numeric, remove NaNs, and convert to integer type
df['min_UGs'] = pd.to_numeric(df['min_UGs'], errors='coerce')
df = df[df['min_UGs'].notna()].copy()
df['min_UGs'] = df['min_UGs'].astype(int)

# Select features to be used in the model
feature_columns = [
    'Vang_XES_0-',
    'Vang_XES_0+',
    'Vang_XES_200ms',
    'Vang_XES_100ms',
    'Vang_RioVT_EST_0-',
    'Vang_RioVT_EST_0+',
    'Vang_RioVT_EST_200ms',
    'Vang_RioVT_EST_100ms'
]

# Prepare feature matrix X and target vector y
X = df[feature_columns].values
y = df['min_UGs'].values

# Split dataset into training and testing sets (80% train, 20% test), stratified by class
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize features (mean=0, std=1) to improve model convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler on training data and transform
X_test_scaled = scaler.transform(X_test)        # Transform test data using the same scaler

# --- Function to calculate and print class-wise metrics (FN, FP, TP, TN) ---
def print_all_metrics_per_class(y_true, y_pred, label):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    results = {}
    for c in classes:
        # False Negative: actual class c but predicted lower class (underestimation)
        FN = np.sum((y_true == c) & (y_pred < y_true))
        # False Positive: actual class c but predicted higher class (overestimation)
        FP = np.sum((y_true == c) & (y_pred > y_true))
        # True Positive: predicted class equals actual class
        TP = np.sum((y_true == c) & (y_pred == y_true))
        # True Negative: samples not belonging to class c and predicted not c
        TN = np.sum((y_true != c) & (y_pred != c))
        results[c] = {'FN': FN, 'FP': FP, 'TP': TP, 'TN': TN}
    print(f"\nMetrics per class ({label}):")
    for c, v in results.items():
        print(f"Class {c}: {v}")

# --- Function to plot PCA results highlighting false negatives and false positives ---
def plot_pca_fn(X_pca, y_true, y_pred, title, ax):
    unique_classes = np.unique(y_true)
    colors = plt.cm.tab10.colors  # Color palette for up to 10 classes
    correct_mask = (y_true == y_pred)  # Mask for correctly classified samples
    # Plot correctly classified points in light gray
    ax.scatter(X_pca[correct_mask, 0], X_pca[correct_mask, 1],
               c='lightgray', label='Correctly Classified', s=60, edgecolor='k', zorder=1)
    # Plot false negatives (X) and false positives (P) per class in specific colors
    for i, cls in enumerate(unique_classes):
        fn_mask = (y_true == cls) & (y_pred < y_true)
        fp_mask = (y_true == cls) & (y_pred > y_true)
        ax.scatter(X_pca[fn_mask, 0], X_pca[fn_mask, 1],
                   marker='X', s=120, color=colors[int(cls) % 10],
                   label=f'FN Class {cls}', edgecolor='k', zorder=2)
        ax.scatter(X_pca[fp_mask, 0], X_pca[fp_mask, 1],
                   marker='P', s=100, color=colors[int(cls) % 10],
                   label=f'FP Class {cls}', edgecolor='k', zorder=3)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)

# --- 1. Train and evaluate OrdinalRidge model on original dataset (no balancing) ---
clf_puro = OrdinalRidge()
clf_puro.fit(X_train_scaled, y_train)
y_train_pred_puro = clf_puro.predict(X_train_scaled)
y_test_pred_puro = clf_puro.predict(X_test_scaled)

# PCA for 2D visualization
pca_puro = PCA(n_components=2)
X_train_pca_puro = pca_puro.fit_transform(X_train_scaled)
X_test_pca_puro = pca_puro.transform(X_test_scaled)

# Create figure with subplots for train and test
fig_puro, axs_puro = plt.subplots(1, 2, figsize=(18, 7))
plot_pca_fn(X_train_pca_puro, y_train, y_train_pred_puro, "Train (Original Dataset)", axs_puro[0])
plot_pca_fn(X_test_pca_puro, y_test, y_test_pred_puro, "Test (Original Dataset)", axs_puro[1])

# Organize legend, title and save figure
handles, labels = [], []
for ax in axs_puro:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
by_label = OrderedDict(zip(labels, handles))
fig_puro.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, fontsize=10)
fig_puro.suptitle("OrdinalRidge – FN (X), FP (P) – Original Dataset", fontsize=16)
fig_puro.tight_layout(rect=[0,0,1,0.93])
fig_path_puro = os.path.join(outdir, "pca_train_test_fn_fp_original_dataset.png")
fig_puro.savefig(fig_path_puro, dpi=300)
plt.close(fig_puro)
print(f"Figure saved: {fig_path_puro}")

# Print detailed metrics for train and test sets
print_all_metrics_per_class(y_train, y_train_pred_puro, 'OrdinalRidge TRAIN original')
print_all_metrics_per_class(y_test, y_test_pred_puro, 'OrdinalRidge TEST original')

# --- 2. Apply SMOTE for class balancing and retrain model ---
min_class_size = min(Counter(y_train).values())  # Find minority class size
k_neighbors = max(1, min_class_size - 1)         # Set parameter for SMOTE
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)  # Oversample minority classes

# Train OrdinalRidge on balanced data
clf_smote = OrdinalRidge()
clf_smote.fit(X_train_res, y_train_res)
y_train_pred_smote = clf_smote.predict(X_train_res)
y_test_pred_smote = clf_smote.predict(X_test_scaled)  # Test on original test data

# PCA for balanced data
pca_smote = PCA(n_components=2)
X_train_res_pca = pca_smote.fit_transform(X_train_res)
X_test_pca_smote = pca_smote.transform(X_test_scaled)

# Visualize post-SMOTE results
fig_smote, axs_smote = plt.subplots(1, 2, figsize=(18, 7))
plot_pca_fn(X_train_res_pca, y_train_res, y_train_pred_smote, "Train (SMOTE)", axs_smote[0])
plot_pca_fn(X_test_pca_smote, y_test, y_test_pred_smote, "Test (SMOTE)", axs_smote[1])

handles, labels = [], []
for ax in axs_smote:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
by_label = OrderedDict(zip(labels, handles))
fig_smote.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, fontsize=10)
fig_smote.suptitle("OrdinalRidge – FN (X), FP (P) – After SMOTE", fontsize=16)
fig_smote.tight_layout(rect=[0,0,1,0.93])
fig_path_smote = os.path.join(outdir, "pca_train_test_fn_fp_smote.png")
fig_smote.savefig(fig_path_smote, dpi=300)
plt.close(fig_smote)
print(f"Figure saved: {fig_path_smote}")

# Print metrics after SMOTE
print_all_metrics_per_class(y_train_res, y_train_pred_smote, 'OrdinalRidge TRAIN SMOTE')
print_all_metrics_per_class(y_test, y_test_pred_smote, 'OrdinalRidge TEST SMOTE')

# --- 3. Train with SMOTE + class weighting and search for best alpha ---
# Initialize equal sample weights
sample_weight = np.ones(len(y_train_res))
# Assign higher weights to non-zero classes to penalize errors more
for cls in np.unique(y_train_res):
    if cls != 0:
        sample_weight[y_train_res == cls] = 2.0

# List of regularization parameters to test
alphas = [0.01, 0.1, 1, 10, 100]
fn_results = []
best_alpha = None
min_fn = None

# Search for alpha minimizing false negatives on train set
for alpha in alphas:
    clf = OrdinalRidge(alpha=alpha)
    clf.fit(X_train_res, y_train_res, sample_weight=sample_weight)
    y_train_pred_tmp = clf.predict(X_train_res)
    fn = np.sum(y_train_pred_tmp < y_train_res)
    fn_results.append(fn)
    print(f"alpha={alpha}, False Negatives on balanced train: {fn}")
    if (min_fn is None) or (fn < min_fn):
        min_fn = fn
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha} (lowest number of FNs: {min_fn})")

# Train final model with best alpha and sample weights
clf_final = OrdinalRidge(alpha=best_alpha)
clf_final.fit(X_train_res, y_train_res, sample_weight=sample_weight)
y_train_pred_ponderado = clf_final.predict(X_train_res)
y_test_pred_ponderado = clf_final.predict(X_test_scaled)

# Visualize final weighted model results
fig_ponderado, axs_ponderado = plt.subplots(1, 2, figsize=(18, 7))
plot_pca_fn(X_train_res_pca, y_train_res, y_train_pred_ponderado, "Train (SMOTE + weighting)", axs_ponderado[0])
plot_pca_fn(X_test_pca_smote, y_test, y_test_pred_ponderado, "Test (SMOTE + weighting)", axs_ponderado[1])

handles, labels = [], []
for ax in axs_ponderado:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
by_label = OrderedDict(zip(labels, handles))
fig_ponderado.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, fontsize=10)
fig_ponderado.suptitle(f"OrdinalRidge (SMOTE, weighted) – FN (X), FP (P) – alpha={best_alpha}", fontsize=16)
fig_ponderado.tight_layout(rect=[0,0,1,0.93])
fig_path_ponderado = os.path.join(outdir, "pca_train_test_fn_fp_smote_weighted.png")
fig_ponderado.savefig(fig_path_ponderado, dpi=300)
plt.close(fig_ponderado)
print(f"Figure saved: {fig_path_ponderado}")

# Print final weighted metrics
print_all_metrics_per_class(y_train_res, y_train_pred_ponderado, 'OrdinalRidge TRAIN SMOTE weighted')
print_all_metrics_per_class(y_test, y_test_pred_ponderado, 'OrdinalRidge TEST SMOTE weighted')

# --- 4. Alpha search function for other mord models (LogisticIT, LogisticAT) ---
alphalist = [0.01, 0.1, 1, 10, 100]

def busca_alpha_mord(ModelClass, X_train, y_train, sample_weight, alphalist):
    min_fn = None
    best_alpha = None
    best_model = None
    for alpha in alphalist:
        clf = ModelClass(alpha=alpha)
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        y_train_pred = clf.predict(X_train)
        fn = np.sum(y_train_pred < y_train)
        print(f"{ModelClass.__name__}, alpha={alpha}, False Negatives on balanced train: {fn}")
        if (min_fn is None) or (fn < min_fn):
            min_fn = fn
            best_alpha = alpha
            best_model = clf
    print(f"Best alpha for {ModelClass.__name__}: {best_alpha} (lowest number of FNs: {min_fn})")
    return best_model, best_alpha

# Train LogisticIT with optimal alpha
clf_logitit, best_alpha_logitit = busca_alpha_mord(
    LogisticIT, X_train_res, y_train_res, sample_weight, alphalist)
y_train_pred_logitit = clf_logitit.predict(X_train_res)
y_test_pred_logitit = clf_logitit.predict(X_test_scaled)

fig_logitit, axs_logitit = plt.subplots(1, 2, figsize=(18, 7))
plot_pca_fn(X_train_res_pca, y_train_res, y_train_pred_logitit, "Train (LogisticIT)", axs_logitit[0])
plot_pca_fn(X_test_pca_smote, y_test, y_test_pred_logitit, "Test (LogisticIT)", axs_logitit[1])

handles, labels = [], []
for ax in axs_logitit:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
by_label = OrderedDict(zip(labels, handles))
fig_logitit.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, fontsize=10)
fig_logitit.suptitle(f"LogisticIT (SMOTE, weighted) – FN (X), FP (P) – alpha={best_alpha_logitit}", fontsize=16)
fig_logitit.tight_layout(rect=[0,0,1,0.93])
fig_path_logitit = os.path.join(outdir, "pca_train_test_fn_fp_logisticit_smote_weighted.png")
fig_logitit.savefig(fig_path_logitit, dpi=300)
plt.close(fig_logitit)
print(f"Figure saved: {fig_path_logitit}")

print_all_metrics_per_class(y_train_res, y_train_pred_logitit, 'LogisticIT TRAIN SMOTE weighted')
print_all_metrics_per_class(y_test, y_test_pred_logitit, 'LogisticIT TEST SMOTE weighted')

# Train LogisticAT with optimal alpha
clf_logitat, best_alpha_logitat = busca_alpha_mord(
    LogisticAT, X_train_res, y_train_res, sample_weight, alphalist)
y_train_pred_logitat = clf_logitat.predict(X_train_res)
y_test_pred_logitat = clf_logitat.predict(X_test_scaled)

fig_logitat, axs_logitat = plt.subplots(1, 2, figsize=(18, 7))
plot_pca_fn(X_train_res_pca, y_train_res, y_train_pred_logitat, "Train (LogisticAT)", axs_logitat[0])
plot_pca_fn(X_test_pca_smote, y_test, y_test_pred_logitat, "Test (LogisticAT)", axs_logitat[1])

handles, labels = [], []
for ax in axs_logitat:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
by_label = OrderedDict(zip(labels, handles))
fig_logitat.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, fontsize=10)
fig_logitat.suptitle(f"LogisticAT (SMOTE, weighted) – FN (X), FP (P) – alpha={best_alpha_logitat}", fontsize=16)
fig_logitat.tight_layout(rect=[0,0,1,0.93])
fig_path_logitat = os.path.join(outdir, "pca_train_test_fn_fp_logisticat_smote_weighted.png")
fig_logitat.savefig(fig_path_logitat, dpi=300)
plt.close(fig_logitat)
print(f"Figure saved: {fig_path_logitat}")

print_all_metrics_per_class(y_train_res, y_train_pred_logitat, 'LogisticAT TRAIN SMOTE weighted')
print_all_metrics_per_class(y_test, y_test_pred_logitat, 'LogisticAT TEST SMOTE weighted')

# Fine tuning LogisticIT with inverse frequency sample weights ---
counts = np.bincount(y_train_res)
weights_per_class = np.sum(counts) / (len(counts) * counts)
sample_weight_opt = weights_per_class[y_train_res]
alphalist = np.logspace(-3, 2, 15)
min_fn = None
best_alpha = None
best_model = None
for alpha in alphalist:
    clf = LogisticIT(alpha=alpha)
    clf.fit(X_train_res, y_train_res, sample_weight=sample_weight_opt)
    y_train_pred = clf.predict(X_train_res)
    fn = np.sum(y_train_pred < y_train_res)
    if (min_fn is None) or (fn < min_fn):
        min_fn = fn
        best_alpha = alpha
        best_model = clf

y_train_pred = best_model.predict(X_train_res)
y_test_pred = best_model.predict(X_test_scaled)

fig_final, axs_final = plt.subplots(1, 2, figsize=(18, 7))
plot_pca_fn(X_train_res_pca, y_train_res, y_train_pred, "Train (LogisticIT optimized)", axs_final[0])
plot_pca_fn(X_test_pca_smote, y_test, y_test_pred, "Test (LogisticIT optimized)", axs_final[1])

handles, labels = [], []
for ax in axs_final:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
fig_final.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, fontsize=10)
fig_final.suptitle(f"LogisticIT (SMOTE, inverse weight, optimized) – FN (X), FP (P) – alpha={best_alpha:.4f}", fontsize=16)
fig_final.tight_layout(rect=[0,0,1,0.93])
fig_path_final = os.path.join(outdir, "pca_train_test_fn_fp_logisticit_optimized.png")
fig_final.savefig(fig_path_final, dpi=300)
plt.close(fig_final)
print(f"Figure saved: {fig_path_final}")

print_all_metrics_per_class(y_train_res, y_train_pred, 'LogisticIT TRAIN SMOTE inverse weighted (optimized)')
print_all_metrics_per_class(y_test, y_test_pred, 'LogisticIT TEST SMOTE inverse weighted (optimized)')
