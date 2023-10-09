# %% [markdown]
# # Predicting medical no-shows with CAV
# ML Project (group 7): Continuous variables (CAV)

# %% [markdown]
# #### Importing necessary libraries and packages

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import classification_report


# %% [markdown]
# #### Setting global random seed (necessary for sklearn models)

# %%
np.random.seed(42)

# %% [markdown]
# #### Loading dataset

# %%
df = pd.read_csv("KaggleV2-May-2016.csv")

# %% [markdown]
# #### Variable manipulation

# %%
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["Gender"] = df["Gender"].map({"F": True, "M": False})
df["No-show"] = df["No-show"].map({"Yes": True, "No": False})
df["Scholarship"] = df["Scholarship"].astype(bool)
df["Diabetes"] = df["Diabetes"].astype(bool)
df["Hipertension"] = df["Hipertension"].astype(bool)
df["Alcoholism"] = df["Alcoholism"].astype(bool)
df["SMS_received"] = df["SMS_received"].astype(bool)
df["handicap_boolean"] = df["Handcap"].replace([2, 3, 4], 1).astype(bool)

# %%
df["only_date_appointment_day"] = df["AppointmentDay"].dt.date
df["only_date_scheduled_day"] = df["ScheduledDay"].dt.date
df["lead_days"] = (
    df["only_date_appointment_day"] - df["only_date_scheduled_day"]
).dt.days.astype(np.int64)
df = df.drop("only_date_appointment_day", axis=1)
df = df.drop("only_date_scheduled_day", axis=1)

# %%
df["Age_0_4"] = df["Age"] <= 4
df["Age_5_12"] = (df["Age"] > 4) & (df["Age"] <= 12)
df["Age_13_19"] = (df["Age"] > 12) & (df["Age"] <= 19)
df["Age_20_29"] = (df["Age"] > 19) & (df["Age"] <= 29)
df["Age_30_39"] = (df["Age"] > 29) & (df["Age"] <= 39)
df["Age_40_49"] = (df["Age"] > 39) & (df["Age"] <= 49)
df["Age_50_59"] = (df["Age"] > 49) & (df["Age"] <= 59)
df["Age_60_69"] = (df["Age"] > 59) & (df["Age"] <= 69)
df["Age_70_79"] = (df["Age"] > 69) & (df["Age"] <= 79)
df["Age_80_plus"] = df["Age"] > 79

df["no_waiting_time"] = df["lead_days"] == 0
df["lead_days_1_2_days"] = (df["lead_days"] == 1) | (df["lead_days"] == 2)
df["lead_days_3_days_1_week"] = (df["lead_days"] >= 3) & (df["lead_days"] <= 7)
df["lead_days_1_week_2_weeks"] = (df["lead_days"] > 7) & (df["lead_days"] <= 14)
df["lead_days_2_weeks_1_month"] = (df["lead_days"] > 14) & (df["lead_days"] <= 30)
df["lead_days_more_than_1_month"] = df["lead_days"] > 30

# %% [markdown]
# ## Data preperation

# %% [markdown]
# #### Data cleaning

# %%
df = df[df["Age"] >= 0]
df = df[df["Age"] <= 100]
df = df[df["lead_days"] >= 0]
df = df[df["lead_days"] != 398]
df = df.drop("SMS_received", axis=1)
df = df.drop("Neighbourhood", axis=1)

# %% [markdown]
# #### Creating train/test-set

# %%
predictors_cav = [
    "Gender",
    "Age_0_4",
    "Age_5_12",
    "Age_13_19",
    "Age_20_29",
    "Age_30_39",
    "Age_40_49",
    "Age_50_59",
    "Age_60_69",
    "Age_70_79",
    "Age_80_plus",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "no_waiting_time",
    "lead_days_1_2_days",
    "lead_days_3_days_1_week",
    "lead_days_1_week_2_weeks",
    "lead_days_2_weeks_1_month",
    "lead_days_more_than_1_month",
    "handicap_boolean",
]
target = "No-show"

X_train, X_test, y_train, y_test = train_test_split(
    df[predictors_cav], df[target], test_size=0.2, random_state=42
)

# %% [markdown]
# #### Data balancing

# %% [markdown]
# Oversampling SMOTE

# %%
sm = SMOTENC(random_state=42, categorical_features=[True])
X_train_smote_cav, y_train_smote_cav = sm.fit_resample(X_train, y_train)

print("Original dataset shape")
print("False:", sum(y_train == False))
print("True: ", sum(y_train == True))
print("Resampled dataset CAV shape with SMOTE")
print("False:", sum(y_train_smote_cav == False))
print("True: ", sum(y_train_smote_cav == True))

# %% [markdown]
# Oversampling - duplicating

# %%
ros = RandomOverSampler(random_state=42)
X_train_ros_cav, y_train_ros_cav = ros.fit_resample(X_train, y_train)

print("Original dataset shape")
print("False:", sum(y_train == False))
print("True: ", sum(y_train == True))
print("Resampled dataset CAV shape with OVERSAMPLING")
print("False:", sum(y_train_ros_cav == False))
print("True: ", sum(y_train_ros_cav == True))

# %% [markdown]
# Undersampling - removing

# %%
rus = RandomUnderSampler(random_state=42)
X_train_rus_cav, y_train_rus_cav = rus.fit_resample(X_train, y_train)

print("Original dataset shape")
print("False:", sum(y_train == False))
print("True: ", sum(y_train == True))
print("Resampled dataset CAV shape with UNDERSAMPLING")
print("False:", sum(y_train_rus_cav == False))
print("True: ", sum(y_train_rus_cav == True))

# %% [markdown]
# #### Feature Selection
# For this we will use the Mutual Information ML technique, as this has been identified as an appropriate feature selection methods for classification problems in which one has many categorical predictor variables. Using default of n_neighbors = 5.


# %%
def identify_features_with_mi_zero(X_train, y_train):
    mi = MIC(X_train, y_train, random_state=42, n_neighbors=5)
    mi = pd.Series(mi)
    mi.index = X_train.columns
    print(mi.sort_values())
    mi = mi.to_frame()
    mi.columns = ["MI Score"]
    mi = mi[mi["MI Score"] != 0]
    print(mi.sort_values(by="MI Score"))


# %%
identify_features_with_mi_zero(X_train_smote_cav, y_train_smote_cav)
identify_features_with_mi_zero(X_train_ros_cav, y_train_ros_cav)
identify_features_with_mi_zero(X_train_rus_cav, y_train_rus_cav)

# %% [markdown]
# ## Modelling

# %% [markdown]
# ### Logistic regression


# %%
def print_and_visulise_confusion_matrix(y_test, y_pred):
    """
    Prints a confusion matrix and visualizes the confusion matrix
    """
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion matrix")
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
    print("Classification report")
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


# %% [markdown]
# #### - Oversampled data - SMOTE

# %%
run_and_evaluate_logistic_regression(
    X_train_smote_cav, y_train_smote_cav, X_test, y_test
)

# %% [markdown]
# #### - Oversampled data - duplicates

# %%
run_and_evaluate_logistic_regression(X_train_ros_cav, y_train_ros_cav, X_test, y_test)

# %% [markdown]
# #### - Undersampled data - removal

# %%
run_and_evaluate_logistic_regression(X_train_rus_cav, y_train_rus_cav, X_test, y_test)

# %% [markdown]
# #### - Unbalanced data

# %% [markdown]
# ### Random Forests


# %%
def print_and_visualize_confusion_matrix_rf(y_test, y_pred):
    """
    Prints a confusion matrix and visualizes the confusion matrix
    """
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion matrix")
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


def run_and_evaluate_random_forest_classifier(X_train, y_train, X_test, y_test):
    max_depth = [3, 4, 5, 6, 7]
    n_estimators = [64, 128, 256]
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

    dfrst = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion="entropy",
        random_state=42,
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


def feature_importance_for_rfc(X_train, y_train, n_estimators, max_depth):
    rfc = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    rfc.fit(X_train, y_train)
    importances = rfc.feature_importances_
    columns = X_train.columns
    i = 0
    co_list = []
    fi_list = []
    df_fi = pd.DataFrame()

    while i < len(columns):
        print(
            f"The importance of feature '{columns[i]}' is {round(importances[i]* 100, 2)}%"
        )
        co_list.append(columns[i])
        fi_list.append(importances[i])
        i += 1

    df_fi["Feature"] = co_list
    df_fi["Importance"] = fi_list
    return df_fi


# %%
run_and_evaluate_random_forest_classifier(
    X_train_smote_cav, y_train_smote_cav, X_test, y_test
)

# %%
run_and_evaluate_random_forest_classifier(
    X_train_ros_cav, y_train_ros_cav, X_test, y_test
)

# %%
run_and_evaluate_random_forest_classifier(
    X_train_rus_cav, y_train_rus_cav, X_test, y_test
)

# %% [markdown]
# #### Analysis of feature importances after the RF classifiers have been run and the best parameters are found

# %%
fi_smote_cav = feature_importance_for_rfc(X_train_smote_cav, y_train_smote_cav, 256, 7)

# %%
fi_ros_cav = feature_importance_for_rfc(X_train_ros_cav, y_train_ros_cav, 256, 7)

# %%
fi_rus_cav = feature_importance_for_rfc(X_train_rus_cav, y_train_rus_cav, 64, 6)


# %%
def feature_importance_plot(list1, list2, list3):
    lista = list1.merge(list2, on="Feature")
    final_list = lista.merge(list3, on="Feature")
    final_list["Average Feature Importance"] = final_list[
        ["Importance_x", "Importance"]
    ].mean(axis=1)

    final_list = final_list.sort_values("Average Feature Importance", ascending=False)
    plot = sns.barplot(
        data=final_list,
        x="Average Feature Importance",
        y="Feature",
        orient="h",
        palette=["#72B7A1"],
    )

    return plot


# %%
feature_importance_plot(fi_smote_cav, fi_ros_cav, fi_rus_cav)

# %% [markdown]
# ### Support Vector Machines


# %%
def print_and_visualize_confusion_matrix(y_test, y_pred):
    """
    Prints a confusion matrix and visualizes the confusion matrix
    """
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion matrix")
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
    Prints classification report based on y_test outputs, real outputs and predicted outputs
    """
    target_names = ["show", "no-show"]
    print("Classification report.")
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


# %%
run_and_evaluate_support_vector_machine(
    X_train_smote_cav, y_train_smote_cav, X_test, y_test
)

# %%
run_and_evaluate_support_vector_machine(
    X_train_ros_cav, y_train_ros_cav, X_test, y_test
)

# %%
run_and_evaluate_support_vector_machine(
    X_train_rus_cav, y_train_rus_cav, X_test, y_test
)
