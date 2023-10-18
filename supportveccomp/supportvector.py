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


def run_and_evaluate_support_vector_machine(X_train, y_train, X_test, y_test):
    """
    Runs and evaluates the support vector machine, will output the confusion matrix,
    its visualisation, the classification report, the auc plot and apply recursive
    feature selection for a given number of features.

    Dual = false because n_samples > n_features and this would otherwise require
    the computation of an n_samples x n_samples matrix
    """
    param_grid_svc = {"C": [0.01, 0.1, 1, 10, 100, 1000]}
    linear_svc = LinearSVC(dual=False, max_iter=1000, random_state=42)
    grid = GridSearchCV(linear_svc, param_grid_svc, refit=True, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_, grid.best_estimator_)
    grid_predict = grid.predict(X_test)
    print_and_visualize_confusion_matrix(y_test, grid_predict)
    print_classification_report(y_test, grid_predict)


run_and_evaluate_support_vector_machine(
    X_train_smote_cov, y_train_smote_cov, X_test, y_test
)