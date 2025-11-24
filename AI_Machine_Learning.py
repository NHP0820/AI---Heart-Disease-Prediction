import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import time
import joblib
import os

# =========================
# preprocess
# =========================

def standardize_fit(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def standardize_transform(X, mu, sigma):
    return (np.asarray(X, dtype=float) - mu) / sigma

def train_test_split_np(X, y, test_size=0.2, stratify=True, random_state=42):
    rng = np.random.RandomState(random_state)
    X = np.asarray(X); y = np.asarray(y)
    idxs = np.arange(len(y))
    if stratify:
        X_tr, X_te, y_tr, y_te = [], [], [], []
        for c in np.unique(y):
            c_idx = idxs[y == c]
            rng.shuffle(c_idx)
            cut = int(len(c_idx) * (1 - test_size))
            tr = c_idx[:cut]; te = c_idx[cut:]
            X_tr.append(X[tr]); y_tr.append(y[tr])
            X_te.append(X[te]); y_te.append(y[te])
        X_train = np.vstack(X_tr); y_train = np.concatenate(y_tr)
        X_test  = np.vstack(X_te); y_test  = np.concatenate(y_te)
        # shuffle final
        sidx = rng.permutation(len(y_train))
        X_train, y_train = X_train[sidx], y_train[sidx]
        sidx = rng.permutation(len(y_test))
        X_test, y_test = X_test[sidx], y_test[sidx]
        return X_train, X_test, y_train, y_test
    else:
        rng.shuffle(idxs)
        cut = int(len(idxs) * (1 - test_size))
        tr = idxs[:cut]; te = idxs[cut:]
        return X[tr], X[te], y[tr], y[te]

def stratified_kfold_indices(y, n_splits=10, shuffle=True, random_state=42):
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    folds = [[] for _ in range(n_splits)]
    for c in np.unique(y):
        c_idx = np.where(y == c)[0].tolist()
        if shuffle: rng.shuffle(c_idx)
        for i, idx in enumerate(c_idx):
            folds[i % n_splits].append(idx)
    # make (train_idx, val_idx)
    all_idx = set(range(len(y)))
    pairs = []
    for i in range(n_splits):
        val_idx = np.array(sorted(folds[i]))
        tr_idx = np.array(sorted(list(all_idx - set(val_idx))))
        pairs.append((tr_idx, val_idx))
    return pairs

# ======== SMOTE）========
def smote_binary(X, y, k=5, random_state=42):
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y  # no binary return
    maj = classes[np.argmax(counts)]
    minc = classes[np.argmin(counts)]
    n_maj = counts.max(); n_min = counts.min()
    if n_min == n_maj:
        return X, y
    need = n_maj - n_min
    X_min = X[y == minc]
    # calculate pairwise distances
    from scipy.spatial.distance import cdist
    D = cdist(X_min, X_min)
    np.fill_diagonal(D, np.inf)
    synth = []
    for _ in range(need):
        i = rng.randint(0, len(X_min))
        nn_idx = np.argsort(D[i])[:k]
        j = rng.choice(nn_idx)
        lam = rng.rand()
        newx = X_min[i] + lam * (X_min[j] - X_min[i])
        synth.append(newx)
    if synth:
        X_new = np.vstack([X, np.vstack(synth)])
        y_new = np.concatenate([y, np.full(len(synth), minc, dtype=int)])
        return X_new, y_new
    return X, y

# =========================
# manual metrics and plots
# =========================
def cm2x2(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp],[fn, tp]]), (tn, fp, fn, tp)

def acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def prec(y_true, y_pred, positive=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == positive) & (y_pred == positive))
    fp = np.sum((y_true != positive) & (y_pred == positive))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

def rec(y_true, y_pred, positive=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == positive) & (y_pred == positive))
    fn = np.sum((y_true == positive) & (y_pred != positive))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

def f1(y_true, y_pred, positive=1):
    p = prec(y_true, y_pred, positive)
    r = rec(y_true, y_pred, positive)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def classification_report_text(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    lines = []
    header = "              precision    recall  f1-score   support"
    lines.append(header)
    for cls in [0, 1]:
        p = prec(y_true, y_pred, positive=cls)
        r = rec(y_true, y_pred, positive=cls)
        f = f1(y_true, y_pred, positive=cls)
        s = int(np.sum(y_true == cls))
        lines.append(f"{cls:>14} {p:10.2f} {r:8.2f} {f:9.2f} {s:10d}")
    overall_acc = acc(y_true, y_pred)
    lines.append("")
    lines.append(f"accuracy            {overall_acc:>24.2f} {len(y_true):10d}")
    pm = 0.5 * (prec(y_true, y_pred, 0) + prec(y_true, y_pred, 1))
    rm = 0.5 * (rec(y_true, y_pred, 0) + rec(y_true, y_pred, 1))
    fm = 0.5 * (f1(y_true, y_pred, 0) + f1(y_true, y_pred, 1))
    sup = len(y_true)
    lines.append(f"{'macro avg':>14} {pm:10.2f} {rm:8.2f} {fm:9.2f} {sup:10d}")
    w0 = np.sum(y_true == 0) / sup if sup else 0
    w1 = np.sum(y_true == 1) / sup if sup else 0
    pw = w0 * prec(y_true, y_pred, 0) + w1 * prec(y_true, y_pred, 1)
    rw = w0 * rec(y_true, y_pred, 0) + w1 * rec(y_true, y_pred, 1)
    fw = w0 * f1(y_true, y_pred, 0) + w1 * f1(y_true, y_pred, 1)
    lines.append(f"{'weighted avg':>14} {pw:10.2f} {rw:8.2f} {fw:9.2f} {sup:10d}")
    return "\n".join(lines)

def roc_curve_manual(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]
    thresholds = np.r_[np.inf, np.unique(y_score_sorted)[::-1], -np.inf]
    P = np.sum(y_true_sorted == 1)
    N = np.sum(y_true_sorted == 0)
    tpr_list, fpr_list = [], []
    tp_cum = np.cumsum(y_true_sorted == 1)
    fp_cum = np.cumsum(y_true_sorted == 0)
    for thr in thresholds:
        idx = np.searchsorted(-y_score_sorted, -thr, side='right')
        tp = tp_cum[idx - 1] if idx > 0 else 0
        fp = fp_cum[idx - 1] if idx > 0 else 0
        tpr_list.append(tp / P if P > 0 else 0.0)
        fpr_list.append(fp / N if N > 0 else 0.0)
    return np.array(fpr_list), np.array(tpr_list), thresholds

def auc_trapz(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.trapz(y, x))

def plot_confusion_matrix(ax, cm, title):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

# =========================
# KNN
# =========================
def knn_predict(X_train, y_train, X_test, k=5):
    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train).astype(int)
    y_pred = []
    for x in X_test:
        d = np.sqrt(((X_train - x)**2).sum(axis=1))
        idx = np.argsort(d)[:k]
        votes, counts = np.unique(y_train[idx], return_counts=True)
        y_pred.append(votes[np.argmax(counts)])
    return np.array(y_pred, dtype=int)

# =========================
# Logistic Regression
# =========================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_train(X, y, lr=0.1, epochs=1000, l2=0.0, random_state=42):
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    n, p = X.shape
    W = rng.randn(p) * 0.01
    b = 0.0
    for _ in range(epochs):
        z = X @ W + b
        yhat = sigmoid(z)
        # gradients
        grad_W = (X.T @ (yhat - y)) / n + l2 * W
        grad_b = (yhat - y).mean()
        W -= lr * grad_W
        b -= lr * grad_b
    return {"W": W, "b": b}

def logistic_predict(X, model):
    X = np.asarray(X, dtype=float)
    z = X @ model["W"] + model["b"]
    proba = sigmoid(z)
    return (proba >= 0.5).astype(int), proba

# =========================
# Manual: SVM (RBF Kernel, Simplified SMO)
# =========================
def rbf_kernel(X1, X2, gamma):
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    X1_sq = np.sum(X1**2, axis=1)[:, None]
    X2_sq = np.sum(X2**2, axis=1)[None, :]
    dist2 = X1_sq + X2_sq - 2 * (X1 @ X2.T)
    return np.exp(-gamma * dist2)

def svm_rbf_train_simplified_smo(X, y, C=0.1, gamma=0.5, tol=1e-3, max_passes=5, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.where(np.asarray(y).astype(int) == 1, 1.0, -1.0)
    n = X.shape[0]
    alphas = np.zeros(n)
    b = 0.0
    K = rbf_kernel(X, X, gamma)
    rng = np.random.RandomState(random_state)

    passes = 0
    while passes < max_passes:
        num_changed = 0
        for i in range(n):
            Ei = (alphas * y) @ K[:, i] + b - y[i]
            if (y[i] * Ei < -tol and alphas[i] < C) or (y[i] * Ei > tol and alphas[i] > 0):
                j = i
                while j == i:
                    j = rng.randint(0, n)
                Ej = (alphas * y) @ K[:, j] + b - y[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if y[i] != y[j]:
                    L = max(0.0, alpha_j_old - alpha_i_old)
                    H = min(C, C + alpha_j_old - alpha_i_old)
                else:
                    L = max(0.0, alpha_i_old + alpha_j_old - C)
                    H = min(C, alpha_i_old + alpha_j_old)
                if L == H:
                    continue

                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j] -= y[j] * (Ei - Ej) / eta
                if alphas[j] > H: alphas[j] = H
                if alphas[j] < L: alphas[j] = L
                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                b1 = b - Ei - y[i]*(alphas[i]-alpha_i_old)*K[i, i] - y[j]*(alphas[j]-alpha_j_old)*K[i, j]
                b2 = b - Ej - y[i]*(alphas[i]-alpha_i_old)*K[i, j] - y[j]*(alphas[j]-alpha_j_old)*K[j, j]
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)

                num_changed += 1
        if num_changed == 0:
            passes += 1
        else:
            passes = 0

    sv = alphas > 1e-8
    return {"alphas": alphas[sv], "Xsv": X[sv], "ysv": y[sv], "b": b, "gamma": gamma}

def svm_rbf_decision(X, model):
    X = np.asarray(X, dtype=float)
    K = rbf_kernel(X, model["Xsv"], model["gamma"])
    scores = K @ (model["alphas"] * model["ysv"]) + model["b"]
    proba = 1.0 / (1.0 + np.exp(-scores))
    y_pred = (scores >= 0).astype(int)
    return y_pred, proba

# =========================
# VIF and PCA
# =========================
def vif_manual(X_scaled, feature_names):
    Xs = np.asarray(X_scaled, dtype=float)
    n, p = Xs.shape
    out_rows = []
    for i in range(p):
        y = Xs[:, i]
        X_others = np.delete(Xs, i, axis=1)
        X_design = np.c_[np.ones((n, 1)), X_others]
        beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
        y_pred = X_design @ beta
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r2) if (1.0 - r2) > 1e-12 else np.inf
        out_rows.append((feature_names[i], vif))
    return pd.DataFrame(out_rows, columns=["feature", "VIF"])

def pca_2d(X):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z2 = Xc @ Vt[:2].T
    return Z2

# =========================
#   Load and preprocess data (with caching)
# =========================
@st.cache_data
def get_clean_scaled_data():
    df = pd.read_csv("heart.csv").drop_duplicates().dropna()

    def is_outlier_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)

    continuous_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    for col in continuous_cols:
        df = df[~is_outlier_iqr(df[col])]

    X = df.drop("target", axis=1).values
    y = df["target"].values

    mu, sigma = standardize_fit(X)
    X_scaled = standardize_transform(X, mu, sigma)

    X_train, X_test, y_train, y_test = train_test_split_np(
        X_scaled, y, test_size=0.2, stratify=True, random_state=42
    )

    X_train_smote, y_train_smote = smote_binary(X_train, y_train, k=5, random_state=42)

    return X_train_smote, X_test, y_train_smote, y_test, df.drop("target", axis=1).columns, (mu, sigma)

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# ========== Sidebar ==========
st.sidebar.title("🫀 Heart Disease Comparison")
st.sidebar.markdown("Upload your dataset and explore model performance.📊")

reference_df = pd.read_csv("heart.csv")
expected_columns = list(reference_df.columns)

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df_test = pd.read_csv(uploaded_file)
        if list(df_test.columns) != expected_columns:
            st.sidebar.error("❌ Invalid dataset! Columns do not match the required heart.csv structure.")
            df = reference_df.copy()
        else:
            df = df_test.copy()
    except Exception as e:
        st.sidebar.error(f"⚠️ Error reading file: {e}")
        df = reference_df.copy()
else:
    df = reference_df.copy()

st.sidebar.write(f"Dataset shape: {df.shape}")

# ========== Tabs ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Preprocessing",
    "🪭 KNN", 
    "👙 Logistic Regression", 
    "👟 SVM", 
    "📊 Model Comparison"
])

with tab1:
    st.header("🔍 Preprocessing: Missing Values, Outlier, Overfitting")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Dataset Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("🧬 Data Types and Unique Values")
    feature_info = pd.DataFrame({
        "Data Type": df.dtypes,
        "Unique Values": df.nunique(),
        "Missing Values": df.isnull().sum()
    })
    st.dataframe(feature_info)

    st.subheader("🔢 Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("🧩 Target Distribution (Pie Chart)")
    target_counts = df['target'].value_counts()
    labels = ['No Heart Disease' if i == 0 else 'Heart Disease' for i in target_counts.index]
    sizes = target_counts.values
    if len(sizes) > 0:
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.warning("⚠️ No data available to display the pie chart.")

    # cleaning
    st.subheader("Clean the Dataset")
    original_rows = df.shape[0]
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()
    missing_count = df.isnull().sum().sum()
    df = df.dropna()

    def is_outlier_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)

    continuous_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    outlier_mask = pd.Series([False] * df.shape[0], index=df.index)
    for col in continuous_cols:
        outlier_mask = outlier_mask | is_outlier_iqr(df[col])

    outlier_count = outlier_mask.sum()
    df = df[~outlier_mask]
    final_rows = df.shape[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Original", original_rows)
    col2.metric("Duplicates", duplicate_count)
    col3.metric("Missing", missing_count)
    col4.metric("Outliers", outlier_count)
    col5.metric("Final Rows", final_rows)

    with st.expander("📄 View Cleaned Data"):
        st.dataframe(df)

    st.subheader("🔁 Multicollinearity Check (VIF)")
    Xtmp = df.drop("target", axis=1).values
    mu, sigma = standardize_fit(Xtmp)
    X_scaled = standardize_transform(Xtmp, mu, sigma)
    vif_data = vif_manual(X_scaled, list(df.drop("target", axis=1).columns))
    st.dataframe(vif_data)

    # SMOTE + Standardization
    st.subheader("Apply SMOTE and Standardization")
    X = df.drop("target", axis=1).values
    y = df["target"].values
    mu, sigma = standardize_fit(X)
    X_scaled = standardize_transform(X, mu, sigma)
    X_bal, y_bal = smote_binary(X_scaled, y, k=5, random_state=42)

    st.subheader("⚖️ Class Distribution Before & After SMOTE")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Before SMOTE")
        fig1, ax1 = plt.subplots()
        sns.countplot(x=y, ax=ax1, palette="pastel")
        st.pyplot(fig1)
    with c2:
        st.markdown("### After SMOTE")
        fig2, ax2 = plt.subplots()
        sns.countplot(x=y_bal, ax=ax2, palette="muted")
        st.pyplot(fig2)

with tab2:
    st.header("🧠 KNN Pipeline")
    # Load and preprocess data
    df_knn = pd.read_csv("heart.csv").drop_duplicates().dropna()
    def is_outlier_iqr(series):
        Q1 = series.quantile(0.25); Q3 = series.quantile(0.75); IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR; upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)
    for c in ['age', 'trestbps', 'thalach', 'oldpeak']:
        df_knn = df_knn[~is_outlier_iqr(df_knn[c])]

    X = df_knn.drop("target", axis=1).values
    y = df_knn["target"].values
    mu, sigma = standardize_fit(X)
    X_scaled = standardize_transform(X, mu, sigma)
    X_bal, y_bal = smote_binary(X_scaled, y, k=5, random_state=42)

    # === find best k） ===
    st.subheader("Find Best K for KNN")
    k_range = range(1, 21)
    cv_pairs = stratified_kfold_indices(y_bal, n_splits=10, shuffle=True, random_state=42)
    cv_scores = []
    for k in k_range:
        scores_k = []
        for tr_idx, va_idx in cv_pairs:
            y_hat = knn_predict(X_bal[tr_idx], y_bal[tr_idx], X_bal[va_idx], k=k)
            scores_k.append(acc(y_bal[va_idx], y_hat))
        cv_scores.append(np.mean(scores_k))
    best_k = list(k_range)[int(np.argmax(cv_scores))]
    best_score = float(np.max(cv_scores))

    fig, ax = plt.subplots()
    ax.plot(list(k_range), cv_scores, marker='o')
    ax.set_xlabel("Number of Neighbors (K)")
    ax.set_ylabel("Cross-Validated Accuracy")
    ax.set_title("KNN Accuracy vs. K")
    ax.grid(True)
    st.pyplot(fig)
    st.success(f"🏆 Best K (for display) = **{best_k}** with Accuracy = **{best_score:.4f}**")

    # === auto choose k ===
    X_train, X_test, y_train, y_test = train_test_split_np(
        X_scaled, y, test_size=0.2, stratify=True, random_state=42
    )
    X_train_smote, y_train_smote = smote_binary(X_train, y_train, k=5, random_state=42)
    y_pred = knn_predict(X_train_smote, y_train_smote, X_test, k=best_k)

    st.subheader("Evaluation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc(y_test, y_pred):.2f}")
    c2.metric("Precision", f"{prec(y_test, y_pred):.2f}")
    c3.metric("Recall", f"{rec(y_test, y_pred):.2f}")
    c4.metric("F1 Score", f"{f1(y_test, y_pred):.2f}")

    with st.expander("📄 Classification Report"):
        st.code(classification_report_text(y_test, y_pred), language="text")

    cm, _ = cm2x2(y_test, y_pred)
    fig, ax = plt.subplots()
    plot_confusion_matrix(ax, cm, "Confusion Matrix - KNN")
    st.pyplot(fig)

with tab3:
    st.header("📈 Logistic Regression Analysis")
    df_lr = pd.read_csv("heart.csv")

    # 1. correlation matrix
    st.subheader("🔍 Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_lr.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    # 2. standardization
    st.subheader("📊 Feature Scaling (Before vs After)")
    features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    original_data = df_lr[features].values
    mu_f, sigma_f = standardize_fit(original_data)
    scaled_df = pd.DataFrame(standardize_transform(original_data, mu_f, sigma_f), columns=features)
    fig_scale, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()
    for i, col in enumerate(features):
        sns.kdeplot(original_data[:, i], label='Before Scaling', ax=axs[i])
        sns.kdeplot(scaled_df[col], label='After Scaling', ax=axs[i])
        axs[i].set_title(col)
        axs[i].legend()
    axs[-1].axis('off')
    st.pyplot(fig_scale)

    # 3. Data Cleaning + Remove Two Columns in LR
    st.subheader("")
    original_rows = df_lr.shape[0]
    df_lr = df_lr.drop_duplicates().dropna()
    def is_outlier_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)
    for c in ['age', 'trestbps', 'thalach', 'oldpeak']:
        df_lr = df_lr[~is_outlier_iqr(df_lr[c])]
    # only drop 'trestbps' and 'restecg' for LR
    cols_to_drop = ['trestbps', 'restecg']
    df_lr = df_lr.drop(columns=[c for c in cols_to_drop if c in df_lr.columns])

    # 4. train/test split + SMOTE + Logistic Regression
    st.subheader("")
    X = df_lr.drop("target", axis=1).values
    y = df_lr["target"].values
    mu, sigma = standardize_fit(X)
    X_scaled = standardize_transform(X, mu, sigma)
    X_train, X_test, y_train, y_test = train_test_split_np(
        X_scaled, y, test_size=0.2, stratify=True, random_state=42
    )
    X_train_smote, y_train_smote = smote_binary(X_train, y_train, k=5, random_state=42)

    lr_model = logistic_train(X_train_smote, y_train_smote, lr=0.1, epochs=1000, l2=0.0, random_state=42)
    y_pred, y_proba = logistic_predict(X_test, lr_model)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc(y_test, y_pred):.2f}")
    c2.metric("Precision", f"{prec(y_test, y_pred):.2f}")
    c3.metric("Recall", f"{rec(y_test, y_pred):.2f}")
    c4.metric("F1 Score", f"{f1(y_test, y_pred):.2f}")

    with st.expander("📄 Classification Report"):
        st.code(classification_report_text(y_test, y_pred), language="text")

    st.header("Logistic Regression Evaluation")
    cm, _ = cm2x2(y_test, y_pred)
    fig, ax = plt.subplots()
    plot_confusion_matrix(ax, cm, "Confusion Matrix - Logistic Regression")
    st.pyplot(fig)

with tab4:
    st.header("🧠 Support Vector Machine (SVM) Classification")

    X_train_smote, X_test, y_train_smote, y_test, feature_names, scaler_params = get_clean_scaled_data()

    # SVM Training and Evaluation
    gamma = 1.0 / X_train_smote.shape[1]

    svm_model = svm_rbf_train_simplified_smo(X_train_smote, y_train_smote, C=0.1, gamma=gamma, tol=1e-3, max_passes=5, random_state=42)
    y_pred, y_score = svm_rbf_decision(X_test, svm_model)

    st.subheader("📊 Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{prec(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{rec(y_test, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1(y_test, y_pred):.2f}")

    st.text("📄 Classification Report")
    st.code(classification_report_text(y_test, y_pred), language='text')

    st.subheader("📉 Confusion Matrix")
    cm, _ = cm2x2(y_test, y_pred)
    fig, ax = plt.subplots()
    plot_confusion_matrix(ax, cm, "Confusion Matrix - SVM")
    st.pyplot(fig)

    st.subheader("🎯 SVM Decision Boundary (PCA Projection)")
    X_pca = pca_2d(X_train_smote)
    y_train_vis = y_train_smote
    gamma_vis = 1.0 / X_pca.shape[1]
    clf_vis = svm_rbf_train_simplified_smo(X_pca, y_train_vis, C=1.0, gamma=gamma_vis, tol=1e-3, max_passes=5, random_state=42)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z, _ = svm_rbf_decision(grid, clf_vis)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train_vis, cmap=plt.cm.coolwarm, edgecolors='k')
    legend_labels = ['No Disease', 'Heart Disease']
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    ax.set_title("SVM Decision Boundary (Training Data in PCA Space)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True)
    st.pyplot(fig)

with tab5:
    st.header("📊 Model Comparison")

    # Unified Data Cleaning for KNN, SVM
    df_cmp = pd.read_csv("heart.csv")
    df_cmp = df_cmp.drop_duplicates().dropna()

    def is_outlier_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)

    for col in ['age', 'trestbps', 'thalach', 'oldpeak']:
        df_cmp = df_cmp[~is_outlier_iqr(df_cmp[col])]

    # ---------- KNN and SVM use all columns ----------
    X_full = df_cmp.drop("target", axis=1).values
    y = df_cmp["target"].values
    mu_full, sigma_full = standardize_fit(X_full)
    X_full_scaled = standardize_transform(X_full, mu_full, sigma_full)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split_np(
        X_full_scaled, y, test_size=0.2, stratify=True, random_state=42
    )
    X_train_full_sm, y_train_full_sm = smote_binary(X_train_full, y_train_full, k=5, random_state=42)

    # 1) KNN 
    y_pred_knn = knn_predict(X_train_full_sm, y_train_full_sm, X_test_full, k=best_k)

    # 3) SVM(RBF, C=0.1)  
    gamma_full = 1.0 / X_train_full_sm.shape[1]
    svm_model_cmp = svm_rbf_train_simplified_smo(X_train_full_sm, y_train_full_sm, C=0.1, gamma=gamma_full, tol=1e-3, max_passes=5, random_state=42)
    y_pred_svm, y_score_svm = svm_rbf_decision(X_test_full, svm_model_cmp)

    
    df_lr_cmp = df_cmp.copy()
    cols_to_drop = ['trestbps', 'restecg']
    df_lr_cmp = df_lr_cmp.drop(columns=[c for c in cols_to_drop if c in df_lr_cmp.columns])
    X_lr = df_lr_cmp.drop("target", axis=1).values
    
    mu_lr, sigma_lr = standardize_fit(X_lr)
    X_lr_scaled = standardize_transform(X_lr, mu_lr, sigma_lr)
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split_np(
        X_lr_scaled, y, test_size=0.2, stratify=True, random_state=42
    )
    X_train_lr_sm, y_train_lr_sm = smote_binary(X_train_lr, y_train_lr, k=5, random_state=42)
    lr_model_cmp = logistic_train(X_train_lr_sm, y_train_lr_sm, lr=0.1, epochs=1000, l2=0.0, random_state=42)
    y_pred_lr, y_proba_lr = logistic_predict(X_test_lr, lr_model_cmp)

    # results table
    results = [
        {"Model": "KNN (k=best_k)", "Accuracy": acc(y_test_full, y_pred_knn), "Precision": prec(y_test_full, y_pred_knn), "Recall": rec(y_test_full, y_pred_knn), "F1 Score": f1(y_test_full, y_pred_knn)},
        {"Model": "Logistic Regression (drop trestbps, restecg)", "Accuracy": acc(y_test_lr, y_pred_lr), "Precision": prec(y_test_lr, y_pred_lr), "Recall": rec(y_test_lr, y_pred_lr), "F1 Score": f1(y_test_lr, y_pred_lr)},
        {"Model": "SVM (C=0.1)", "Accuracy": acc(y_test_full, y_pred_svm), "Precision": prec(y_test_full, y_pred_svm), "Recall": rec(y_test_full, y_pred_svm), "F1 Score": f1(y_test_full, y_pred_svm)},
    ]
    df_results = pd.DataFrame(results)

    st.dataframe(df_results.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}"
    }))

    st.subheader("🔍 Metric Comparison")
    df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title("Model Performance Comparison")
    plt.ylim(0.75, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)

    # ROC
    st.subheader("📈 ROC Curves")
    plt.figure(figsize=(8, 6))
    # KNN：use 0/1 predictions
    fpr, tpr, _ = roc_curve_manual(y_test_full, y_pred_knn.astype(float))
    plt.plot(fpr, tpr, label=f"KNN (AUC = {auc_trapz(fpr, tpr):.2f})")
    # LR：use probabilities
    fpr, tpr, _ = roc_curve_manual(y_test_lr, y_proba_lr)
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_trapz(fpr, tpr):.2f})")
    # SVM：use scores (post-sigmoid probabilities)
    fpr, tpr, _ = roc_curve_manual(y_test_full, y_score_svm)
    plt.plot(fpr, tpr, label=f"SVM (AUC = {auc_trapz(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot(plt.gcf())

    st.subheader("🧮 Confusion Matrices")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cms = [
        ("KNN (k=best_k)", y_test_full, y_pred_knn),
        ("Logistic Regression (drop trestbps, restecg)", y_test_lr, y_pred_lr),
        ("SVM (C=0.1)", y_test_full, y_pred_svm),
    ]
    for ax, (name, yt, yhat) in zip(axes, cms):
        cm, _ = cm2x2(yt, yhat)
        plot_confusion_matrix(ax, cm, name)
    plt.tight_layout()
    st.pyplot(fig)

    # Show a set of TP/FP/TN/FN (using SVM as an example)
    cm, (tn, fp, fn, tp) = cm2x2(y_test_full, y_pred_svm)
    st.write(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    st.subheader("🥇 Best Performing Model")
    best_row = df_results.loc[df_results["F1 Score"].idxmax()]
    st.success(f"**{best_row['Model']}** performed best with F1 Score: **{best_row['F1 Score']:.3f}**")

    # model size and inference time
    fitted = {
        "KNN (k=best_k)": {"k": 13, "X": X_train_full_sm, "y": y_train_full_sm, "scaler": (mu_full, sigma_full)},
        "Logistic Regression (drop trestbps, restecg)": {"W": lr_model_cmp["W"], "b": lr_model_cmp["b"], "scaler": (mu_lr, sigma_lr)} if 'lr_model_cmp' in locals() else {"W": lr_model["W"], "b": lr_model["b"], "scaler": (mu_lr, sigma_lr)},
        "SVM (C=0.1)": svm_model_cmp
    }
    model_sizes = {}
    inference_times = {}
    for name, model in fitted.items():
        filename = f"{name.replace(' ', '_')}.joblib"
        joblib.dump(model, filename)
        model_sizes[name] = os.path.getsize(filename) / 1024  # KB
        start = time.time()
        # simple inference on a small batch
        if name.startswith("KNN"):
            sample = X_test_full[:min(100, len(X_test_full))]
            _ = knn_predict(X_train_full_sm, y_train_full_sm, sample, k=best_k)
        elif name.startswith("Logistic"):
            sample = X_test_lr[:min(100, len(X_test_lr))]
            _ = logistic_predict(sample, lr_model_cmp)[0]
        else:
            sample = X_test_full[:min(100, len(X_test_full))]
            _ = svm_rbf_decision(sample, svm_model_cmp)[0]
        inference_times[name] = (time.time() - start) * 1000  # ms

    df_meta = pd.DataFrame({
        "Model": list(fitted.keys()),
        "Model Size (KB)": [model_sizes[m] for m in fitted.keys()],
        "Inference Time (ms)": [inference_times[m] for m in fitted.keys()]
    })

    st.subheader("⚙️ Model Efficiency")
    st.dataframe(df_meta.style.format({
        "Model Size (KB)": "{:.2f}",
        "Inference Time (ms)": "{:.2f}"
    }))

    st.subheader("⬇️Download")
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_results)
    st.download_button("📥 Download Model Metrics", data=csv, file_name='model_comparison.csv', mime='text/csv')
