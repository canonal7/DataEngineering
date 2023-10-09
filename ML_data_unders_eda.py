# %% [markdown]
# # Prediciting medical no-shows: Data Undertanding & EDA
# ML Project (group 7)

# %% [markdown]
# ### Importing necessary libraries and packages

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ### Loading dataset

# %%
df = pd.read_csv("KaggleV2-May-2016.csv")

# %% [markdown]
# ### Exploratory data analysis

# %%
print(df.head())

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# Based on the head of the dataset, patientID, appointmentID, ScheduledDay and AppointmentDay are not interesting to see the unique value counts of and they were therefore ignored before look at the value counts.
#

# %%
columns_useful = df.drop(
    ["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"], axis=1
).columns
for column in columns_useful:
    print(column)
    print(df[column].value_counts().sort_index())

# %% [markdown]
# ### Variable manipulation
# To be able to do an exploratory data analysis, some data types had to be changed. fter the inital EDA, it was found that lead_days could be a useful varible. We created two new helper variables only_date_appointment_day and onlye_date_scheduled_day to help determine the lead_days, those helper variables are not considered for the final model.

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

# %% [markdown]
# #### Helper functions

# %%
sns.set_palette("Set2")


def draw_plot_nonbinary(variable):
    sns.histplot(df, x=variable, kde=False)


def draw_plot_binary(variable, left_axis, right_axis, title):
    counts = df.groupby(variable)["No-show"].mean() * 100
    sns.countplot(data=df, x=variable, hue="No-show")
    plt.title(title)
    plt.xlabel(variable)
    plt.ylabel("Number of appointments")
    plt.gca().set_xticklabels([left_axis, right_axis])
    for i in range(len(counts)):
        plt.text(
            i,
            df["No-show"][df[variable] == counts.index[i]].value_counts()[1],
            f"{counts[i]:.2f}%",
            ha="left",
        )
    plt.show()

    sns.set_palette("Set2")


# %% [markdown]
# **Gender**

# %%
gender_distribution = df["Gender"].value_counts(normalize=True) * 100
print("Percentage", gender_distribution)
draw_plot_binary("Gender", "Male", "Female", "No show by gender")

# %% [markdown]
# **Age**

# %%
plt.title("Age distribution")
sns.boxplot(x=df["Age"])

# %%
plt.hist(df["Age"], bins=range(0, 120, 3), edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Number of appointments in age range")
plt.title("Age distribution")
plt.show()

# %% [markdown]
# **Scholarship**

# %%
draw_plot_binary(
    "Scholarship", "No scholarship", "Scholarship", "No show by scholarship status"
)

# %% [markdown]
# **Handicap**

# %%
print(df["Handcap"].value_counts())

# %% [markdown]
# Because of the low number of instances that have more than 1 handicap, we chose to count handicap as a boolean.
#

# %% [markdown]
# **Handicap_boolean, Alocholism, Hypertension and Diabetes**

# %%
draw_plot_binary(
    "Alcoholism", "No alcoholism", "Alcoholism", "No show by alocholism diagnosis"
)
draw_plot_binary(
    "Hipertension",
    "No hypertension",
    "Hypertension",
    "No show by hypertension diagnosis",
)
draw_plot_binary("Diabetes", "No diabetes", "Diabetes", "No show by diabetes diagnosis")
draw_plot_binary(
    "handicap_boolean",
    "No handicap",
    "At least one handicap",
    "No show by handicap diagnosis",
)

# %% [markdown]
# **SMS_received**

# %%
draw_plot_binary("SMS_received", "No SMS", "SMS", "No show by SMS status")

# %% [markdown]
# **Neighborhood_analysis**

# %% [markdown]
# The neigheirhoods no-show count was exported, so that the data could be analysed using QGIS(Geographic Information Systems software). The fourth line is commented out so that no file will be generated everytime this code is run.

# %%
neighborhood_df = df.groupby("Neighbourhood").mean("No-show")
neighborhood_df = neighborhood_df[["No-show"]]
print(neighborhood_df.sort_values("No-show"))
# neighborhood_df.to_csv('neighbourhood_no_show.csv')

# %% [markdown]
# **Lead days**

# %%
print(df["lead_days"].value_counts().nlargest(20))
print(df["lead_days"].sort_values())
print(df.info())

# %%
plt.hist(df["lead_days"], bins=range(0, 120, 3), edgecolor="black")
plt.ylabel("Number of appointments in lead days range")
plt.xlabel("lead_days")
plt.title("Lead days distribution")
plt.show()

# %% [markdown]
# #### Correlation matrix
# This correlation matrix is with the inital variable set

# %%
correlation = df.corr().round(2)
plt.figure(figsize=(14, 7))
sns.heatmap(correlation, annot=True, cmap="YlOrBr")

# %% [markdown]
# #### Age and lead days as categorical variables

# %% [markdown]
# Various age categories have been considered as the literature review revealed that age can be a strong predictor in relation to certain age groups, rather than age as a whole.

# %% [markdown]
# **PatientID and AppointmentID, appointment history**

# %%
not_unique_patient_id = df["PatientId"][df["PatientId"].duplicated()]
df["previous_app"] = df["PatientId"].isin(not_unique_patient_id).astype(int)
print(df["previous_app"].value_counts())

print("Number of unique PatientId values:", df["PatientId"].nunique())
print("Number of unique AppointmentId values:", df["AppointmentID"].nunique())
print(df["PatientId"].value_counts())

df2 = df.groupby("PatientId")["AppointmentID"].count().reset_index()
df2.columns = ["PatientId", "number_of_appointments"]
print(df2.sort_values("number_of_appointments", ascending=False))

df3 = df.groupby("PatientId")["No-show"].mean().reset_index()
df3.columns = ["PatientId", "average_no_show"]
print(df3.sort_values("average_no_show"))

merge1 = pd.merge(df, df2, how="left", on="PatientId")
merge2 = pd.merge(merge1, df3, how="left", on="PatientId")
merge2["no_prev_appointment"] = merge2["number_of_appointments"] == 1
merge2["prev_appointment_show"] = (merge2["number_of_appointments"] > 1) & (
    merge2["average_no_show"] == 0
)
merge2["prev_appointment_no_show"] = (merge2["number_of_appointments"] > 1) & (
    merge2["average_no_show"] != 0
)
df = merge2

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

# %% [markdown]
# For the variable lead_days, different variable types and categories were considered similar to the age variable. EmotionFurther justification has been mentioned in the report.

# %%
df["no_waiting_time"] = df["lead_days"] == 0
df["lead_days_1_2_days"] = (df["lead_days"] == 1) | (df["lead_days"] == 2)
df["lead_days_3_days_1_week"] = (df["lead_days"] >= 3) & (df["lead_days"] <= 7)
df["lead_days_1_week_2_weeks"] = (df["lead_days"] > 7) & (df["lead_days"] <= 14)
df["lead_days_2_weeks_1_month"] = (df["lead_days"] > 14) & (df["lead_days"] <= 30)
df["lead_days_more_than_1_month"] = df["lead_days"] > 30

# %% [markdown]
# #### Better understanding the SMS_received variable

# %% [markdown]
# Patterns within the SMS_received variable were analysed with the aim of finding out if there were confounding effect relating to the SMS_received variable.

# %%
df_sms = df[df["SMS_received"] == True]
df_no_sms = df[df["SMS_received"] == False]
print(
    df_sms[["SMS_received", "AppointmentDay", "ScheduledDay", "lead_days"]]
    .sort_values("AppointmentDay")
    .head(2)
)
print(
    df_no_sms[["SMS_received", "AppointmentDay", "ScheduledDay", "lead_days"]]
    .sort_values("AppointmentDay")
    .head(2)
)
print(
    df_sms[["SMS_received", "AppointmentDay", "ScheduledDay", "lead_days"]]
    .sort_values("ScheduledDay")
    .head(2)
)
print(
    df_no_sms[["SMS_received", "AppointmentDay", "ScheduledDay", "lead_days"]]
    .sort_values("ScheduledDay")
    .head(2)
)
df_no_sms_lead_days = df_no_sms[df_no_sms["lead_days"] > 2]
print(df_no_sms_lead_days.info())
print(
    df_no_sms_lead_days[["SMS_received", "AppointmentDay", "ScheduledDay", "lead_days"]]
    .sort_values("AppointmentDay")
    .head(10)
)
print(
    df_no_sms_lead_days[["SMS_received", "AppointmentDay", "ScheduledDay", "lead_days"]]
    .sort_values("ScheduledDay")
    .head(10)
)

print(df.groupby("Neighbourhood").mean("SMS_received"))
df_3days = df[df["lead_days"] > 2]
print(
    df_3days.groupby("Neighbourhood").mean("SMS_received").sort_values("SMS_received")
)

# %% [markdown]
# #### Final correlation matrix
# This correlation matrix includes age and lead days as  categorical and as continuous variables.

# %%
correlation = df.corr().round(2)
plt.figure(figsize=(20, 16))
sns.heatmap(correlation, annot=True, cmap="YlOrBr")
