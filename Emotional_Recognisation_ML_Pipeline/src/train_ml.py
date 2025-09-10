import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils import load_embeddings

def train_and_evaluate(X, y, model, model_name, results_dir):
    """
    Train ML model and save classification report.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, digits=4)
    print(f"\n[RESULT] {model_name}\n{report}")

    # Save report to file
    report_path = os.path.join(results_dir, f"{model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[SAVED] Report saved to {report_path}")

def main(npz_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    X, y, _ = load_embeddings(npz_path)

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42)
    }

    for name, model in models.items():
        train_and_evaluate(X, y, model, name, results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/embeddings_crema_test_fused.npz")
    parser.add_argument("--results", type=str, default="../results/reports")
    args = parser.parse_args()

    main(args.data, args.results)
