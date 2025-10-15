from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def split_data(df: pd.DataFrame):
    y = df["is_canceled"]
    X = df.drop("is_canceled", axis=1)
    return train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = None
    print(classification_report(y_test, y_pred, digits=4))
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print("ROC AUC (probabilities):", auc)
    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    # if we have probabilities plot ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
        roc_disp.plot()
    plt.show()

def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("=== Random Forest ===")
    evaluate_model(model, X_test, y_test)
    return model

def decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    print("=== Decision Tree ===")
    evaluate_model(model, X_test, y_test)
    return model

def catboost_model(X_train, X_test, y_train, y_test):
    model = CatBoostClassifier(iterations=200, depth=8, verbose=10)
    model.fit(X_train, y_train)
    print("=== CatBoost ===")
    evaluate_model(model, X_test, y_test)
    return model

def xgboost_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier(booster='gbtree', learning_rate=0.1, max_depth=5, n_estimators=180)
    model.fit(X_train, y_train)
    print("=== XGBoost ===")
    evaluate_model(model, X_test, y_test)
    return model

def lightgbm_model(X_train, X_test, y_train, y_test):
    # grid search for best params
    param_grid = {
        'num_leaves': [31, 50, 70],
        'learning_rate': [0.1, 0.05, 0.01],
        'n_estimators': [200, 500, 1000]
    }
    grid = GridSearchCV(LGBMClassifier(random_state=42), param_grid,
                        scoring='roc_auc', cv=3, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("=== LightGBM (best) ===")
    evaluate_model(best, X_test, y_test)
    return best
