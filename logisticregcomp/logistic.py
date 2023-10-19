import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd



def print_and_visulise_confusion_matrix(y_test, y_pred):
    """
    Prints a confusion matrix and visualizes the confusion matrix
    """
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("This is the confusion matrix for the initial regression model")
    print(cnf_matrix)
    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title("Confusion matrix", y=1.1)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")


def print_classification_report(y_test, y_pred):
    """
    Prints classification report based on y_test outputs, real outputs and y_pred outputs
    """
    target_names = ["show", "no-show"]
    print("This is the classification report for the inital regression model.")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))


def run_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Runs and evaluates the logistic regression, will output the confusion matrix,
    its visualisation, the classification report, the auc plot and apply recursive
    feature selection for a given number of features. Hyperparamter tuning included.
    """
    log_reg_params = {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }
    grid_log_reg = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000, solver="liblinear"),
        log_reg_params,
        cv=5,
    )
    grid_log_reg.fit(X_train, y_train)
    logreg = grid_log_reg.best_estimator_
    y_pred = logreg.predict(X_test)
    print_and_visulise_confusion_matrix(y_test, y_pred)
    print_classification_report(y_test, y_pred)

run_and_evaluate_logistic_regression(
    X_train_smote_cov, y_train_smote_cov, X_test, y_test
)