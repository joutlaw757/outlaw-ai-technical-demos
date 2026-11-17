import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def false_positive_rate(y_true, y_pred):
    negatives = y_true == 0
    return float(np.mean(y_pred[negatives] == 1)) if negatives.any() else float('nan')

def main():
    X, y = make_classification(n_samples=1200, n_features=8, n_informative=5,
                               n_redundant=1, random_state=42)
    rng = np.random.default_rng(42)
    # Synthetic grouping for illustration, not a demographic proxy.
    group = rng.integers(0, 2, size=len(y))
    X_train, X_test, y_train, y_test, _, group_test = train_test_split(
        X, y, group, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    pred = model.predict(X_test)
    print(classification_report(y_test, pred, zero_division=0))
    print('Confusion matrix:\n', confusion_matrix(y_test, pred))
    for g in (0, 1):
        mask = group_test == g
        print(f'Synthetic group {g} FPR: {false_positive_rate(y_test[mask], pred[mask]):.3f}')

if __name__ == '__main__':
    main()
