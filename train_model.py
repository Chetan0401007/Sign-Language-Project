import pandas as pd
import numpy as np
import joblib
import os
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network    import MLPClassifier
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.model_selection   import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import (accuracy_score, classification_report,
                                        confusion_matrix, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE      = "data.csv"
MODEL_FILE     = "model.joblib"
LABEL_FILE     = "label_classes.joblib"
REPORT_DIR     = "training_report"
TEST_SIZE      = 0.20
RANDOM_STATE   = 42

def banner(msg: str):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)

def ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)

def load_data():
    banner("Step 1 — Loading dataset")
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        raise FileNotFoundError(
            f"'{DATA_FILE}' is empty or does not exist. Please run 'python collect_data.py' first to collect gesture samples!")

    num_landmarks = 21
    feature_cols  = [f"x{i}" for i in range(num_landmarks)] + \
                    [f"y{i}" for i in range(num_landmarks)] + \
                    [f"z{i}" for i in range(num_landmarks)]
    all_cols      = feature_cols + ["label"]

    try:
        with open(DATA_FILE, "r") as f:
            first_non_comment = ""
            for line in f:
                if line.strip() and not line.startswith("#"):
                    first_non_comment = line
                    break

        has_header = "label" in first_non_comment or "x0" in first_non_comment

        if has_header:
            df = pd.read_csv(DATA_FILE, comment="#")
        else:
            df = pd.read_csv(DATA_FILE, comment="#", header=None, names=all_cols)
    except Exception as e:
        raise ValueError(f"Could not parse '{DATA_FILE}': {e}")

    if df.empty or "label" not in df.columns:
        raise ValueError(f"'{DATA_FILE}' contains no valid gesture records yet.")

    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"  Classes found ({len(df['label'].unique())}): {sorted(df['label'].unique().tolist())}")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}")

    before = len(df)
    df.dropna(inplace=True)
    after  = len(df)
    if before != after:
        print(f"  [!] Dropped {before - after} rows with NaN values.")

    min_class_count = df['label'].value_counts().min()
    if min_class_count < 5:
        raise ValueError(
            f"Class with only {min_class_count} sample(s) detected. "
            "Collect more data (≥ 50 samples per class recommended).")

    return df

def preprocess(df: pd.DataFrame):
    banner("Step 2 — Preprocessing")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Label mapping: { {cls: i for i, cls in enumerate(le.classes_)} }")

    return X, y, le

def split_data(X, y):
    banner("Step 3 — Stratified Train / Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test  samples: {len(X_test):,}")
    return X_train, X_test, y_train, y_test

def build_pipeline():
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
        verbose=False)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    mlp)
    ])
    return pipeline

def cross_validate(pipeline, X_train, y_train):
    banner("Step 4 — 5-Fold Cross Validation (on training set)")
    t0     = time.time()
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=5, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
    print(f"  Fold scores: {[f'{s*100:.2f}%' for s in scores]}")
    print(f"  Time: {time.time() - t0:.1f}s")
    return scores.mean()

def train_final(pipeline, X_train, y_train):
    banner("Step 5 — Training Final Model on Full Training Set")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"  Training complete in {time.time() - t0:.1f}s")
    return pipeline

def evaluate(pipeline, X_test, y_test, le, X_train, y_train):
    banner("Step 6 — Evaluation")
    ensure_report_dir()

    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"  Test Accuracy : {acc*100:.4f}%")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_))

    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(le.classes_)),
                                    max(6, len(le.classes_))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                ax=ax)
    ax.set_title("Confusion Matrix — Sign Language Classifier", fontsize=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {cm_path}")

    mlp = pipeline.named_steps["clf"]
    if hasattr(mlp, "loss_curve_"):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(mlp.loss_curve_, color="#4C72B0", linewidth=2, label="Training Loss")
        if hasattr(mlp, "validation_scores_"):
            val_loss = [1 - s for s in mlp.validation_scores_]
            ax.plot(val_loss, color="#DD8452", linewidth=2,
                    linestyle="--", label="Validation Loss (approx)")
        ax.set_title("MLP Training Loss Curve", fontsize=13)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        lc_path = os.path.join(REPORT_DIR, "loss_curve.png")
        plt.savefig(lc_path, dpi=150)
        plt.close()
        print(f"  Loss curve saved -> {lc_path}")

    return acc

def benchmark_random_forest(X_train, y_train, X_test, y_test):
    banner("Step 7 - Benchmark: Random Forest (comparison)")
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            random_state=RANDOM_STATE))
    ])
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"  Random Forest Test Accuracy: {rf_acc*100:.4f}%")
    return rf_acc

def save_artifacts(pipeline, le):
    banner("Step 8 - Saving Model Artifacts")
    joblib.dump(pipeline, MODEL_FILE, compress=3)
    print(f"  Model saved -> {MODEL_FILE}")

    joblib.dump(le, LABEL_FILE, compress=3)
    print(f"  LabelEncoder saved -> {LABEL_FILE}")

def main():
    print("\n  Sign Language Recognition - Model Trainer  ")
    print("  ============================================\n")

    df                          = load_data()
    X, y, le                    = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline                    = build_pipeline()
    cv_mean                     = cross_validate(pipeline, X_train, y_train)
    pipeline                    = train_final(pipeline, X_train, y_train)
    test_acc                    = evaluate(pipeline, X_test, y_test, le,
                                            X_train, y_train)
    rf_acc                      = benchmark_random_forest(
                                    X_train, y_train, X_test, y_test)
    save_artifacts(pipeline, le)

    banner("Training Summary")
    print(f"  Cross-Val (5-fold) : {cv_mean*100:.2f}%")
    print(f"  MLP  Test Accuracy : {test_acc*100:.4f}%")
    print(f"  RF   Test Accuracy : {rf_acc*100:.4f}%")
    chosen = "MLP" if test_acc >= rf_acc else "Random Forest"
    print(f"  -> Best model used : {chosen}")
    print(f"\n  Run  python predict.py  to launch live prediction!\n")


if __name__ == "__main__":
    main()
