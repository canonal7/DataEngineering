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


def run_and_evaluate_random_forest_classifier(X_train, y_train, X_test, y_test):
    max_depth = [3, 4, 5, 6, 7]
    n_estimators = [64, 128, 256]
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

    dfrst = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    grid = GridSearchCV(estimator=dfrst, param_grid=param_grid, cv=5)
    grid_results = grid.fit(X_train, y_train)

    print(
        "Best: {0}, using {1}".format(
            grid_results.cv_results_["mean_test_score"], grid_results.best_params_
        )
    )
    best_clf = grid_results.best_estimator_
    y_pred = best_clf.predict(X_test)
    target_names = ["show", "no-show"]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))
    print_and_visualize_confusion_matrix_rf(y_test, y_pred)


run_and_evaluate_random_forest_classifier(
    X_train_smote_cov, y_train_smote_cov, X_test, y_test
)