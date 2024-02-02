# -*- coding: utf-8 -*-
"""DOS-Attack-Types-ML-KNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17Yvf4o5KKTE7rNP91wwpU5olmmQNBeFw
"""

from google.colab import drive
drive.mount('/content/drive')

# import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
#!pip install tabulate
from tabulate import tabulate
import os

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

# This function takes a directory path as input and returns a list of CSV files within that directory.
def find_csv_files(directory):
    csv_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    return csv_files

# Provide the directory path where you want to start searching for CSV files
starting_directory = "/content/drive/MyDrive/BoT-IoT Dataset/DoS"

csv_files = find_csv_files(starting_directory)

for csv_file in csv_files:
    print(csv_file.split("/")[-1], "\t\t", csv_file)

df_dos_UDP = pd.read_csv("/content/drive/MyDrive/BoT-IoT Dataset/DoS/DoS_UDP/DoS_UDP[1].csv") # read the UDP data
df_dos_HTTP = pd.read_csv("/content/drive/MyDrive/BoT-IoT Dataset/DoS/DoS_HTTP/DoS_HTTP[1].csv") # read the HTTP data
df_dos_TCP = pd.read_csv("/content/drive/MyDrive/BoT-IoT Dataset/DoS/DoS_TCP/DoS_TCP[1].csv") # read the TCP data

df_dos_UDP.head() # lets show the first 5 rows of dataset

df_dos_HTTP.head() # lets show the first 5 rows of dataset

df_dos_TCP.head() # lets show the first 5 rows of dataset

df_dos_TCP.shape, df_dos_UDP.shape, df_dos_HTTP.shape # check the shape of dataset

df_dos_UDP['Sub_Cat'].value_counts() # check the sub cat value counts of UDP

df_dos_TCP['Sub_Cat'].value_counts() # check the sub cat value counts of TCP

df_dos_HTTP['Sub_Cat'].value_counts() # check the sub cat value counts of HTTP

# only get the Dos Attacks
nrows = 52466  # Define the maximum number of rows to retain.

# Filter the 'df_dos_TCP' DataFrame to include only rows with the 'Sub_Cat' column value 'DoS_TCP' and limit the rows to 'nrows'.
df_dos_TCP = df_dos_TCP[df_dos_TCP['Sub_Cat'] == 'DoS_TCP'][:nrows]

# Filter the 'df_dos_UDP' DataFrame to include only rows with the 'Sub_Cat' column value 'DoS_UDP' and limit the rows to 'nrows'.
df_dos_UDP = df_dos_UDP[df_dos_UDP['Sub_Cat'] == 'DoS_UDP'][:nrows]

# Filter the 'df_dos_HTTP' DataFrame to include only rows with the 'Sub_Cat' column value 'DoS_HTTP' and limit the rows to 'nrows'.
df_dos_HTTP = df_dos_HTTP[df_dos_HTTP['Sub_Cat'] == 'DoS_HTTP'][:nrows]

df_dos_TCP.shape, df_dos_UDP.shape, df_dos_HTTP.shape # now again the check the shape

# remove the irrelevent features from all dataset
df_dos_TCP_ = df_dos_TCP.drop(['Timestamp','Label', 'Cat', 'Flow_ID'], axis = 1)
df_dos_UDP_ = df_dos_UDP.drop(['Timestamp','Label', 'Cat', 'Flow_ID'], axis = 1)
df_dos_HTTP_ = df_dos_HTTP.drop(['Timestamp','Label', 'Cat', 'Flow_ID'], axis = 1)

df_dos_TCP_.head()

# rename the column name and change the class label from all dataset
df_dos_TCP_  = df_dos_TCP_.rename(columns={'Sub_Cat':'class'})
df_dos_TCP_['class']  = df_dos_TCP_['class'].map({'DoS_TCP':'TCP_FLOOD'})

df_dos_UDP_  = df_dos_UDP_.rename(columns={'Sub_Cat':'class'})
df_dos_UDP_['class']  = df_dos_UDP_['class'].map({'DoS_UDP':'UDP_FLOOD'})

df_dos_HTTP_  = df_dos_HTTP_.rename(columns={'Sub_Cat':'class'})
df_dos_HTTP_['class']  = df_dos_HTTP_['class'].map({'DoS_HTTP':'HTTP_FLOOD'})

df = pd.concat([df_dos_TCP_, df_dos_UDP_, df_dos_HTTP_]) # concate all the dataset into one dataset
df.head()

# lets try to check the general information of all columns in dataset
df.info(verbose = True)

"""* The info() function provides essential information about the DataFrame, including the data types, non-null counts, and memory usage of each column. Setting verbose to True enables a detailed output with column-wise information. This summary is useful for understanding the composition and structure of the dataset, identifying any missing values, and gauging memory usage."""

# lets try to check the total missing values in the dataset
print("Total Missing Values : ", df.isnull().sum().sum())
print("Missing Value feature: ", df.columns[df.isnull().any()])

# lets drop the all null occurence
df = df.dropna()

# lets again check the missing values
print("After Removing Total Missing Values : ", df.isnull().sum().sum())

def class_distribution(df_final, col):
    plt.figure(figsize=(17, 7))

    plt.subplot(1, 2, 1)

    colors = ['red', 'green', 'blue', 'magenta', 'orange', 'purple', 'cyan', 'yellow', 'brown']
    ax = df_final[col].value_counts().plot(kind='bar', color=colors)

    plt.xlabel('Category', fontsize=16)
    plt.ylabel('Frequency of Class', fontsize=16)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title(f'Frequency Distribution of Class', fontsize=18)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.01), size=15)

    plt.subplot(1, 2, 2)

    df_final[col].value_counts().plot.pie(autopct='%1.2f%%', shadow=True, colors=colors,
                                          textprops={'fontsize': 15, 'color': 'white'})
    plt.ylabel('Target', fontsize=16)
    plt.title(f'Proportional Distribution of Class', fontsize=18)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()

class_distribution(df, 'class')

"""* Evidently, the dataset exhibits a balanced distribution across its categories.
* The advantage of a balanced dataset is that accuracy evaluation can provide reliable insights. In cases of imbalanced datasets, additional metrics become necessary for accurate assessment.
* To comprehensively assess model performance, we will employ various performance metrics on the testing data, including Precision, Recall, F1-score, and AUC score.
* When dealing with imbalanced data, metrics like ROC/AUC demonstrate enhanced efficacy in evaluating model performance.
"""

# lets remove the features which has only one category value greater 97% because that are not good for model predictiom
stats = []
for col in df.columns[:-1]:
    stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))
stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', '% MissingValues', '% highOneCategoricalValues', 'type'])
df_ = stats_df.sort_values('% MissingValues', ascending=False)
one_category_value = df_[df_['% highOneCategoricalValues']>=97]
print(tabulate(one_category_value, headers = 'keys', tablefmt = 'psql'))

1000 - 0:990 1:10

drop_features = one_category_value['Feature'].tolist()
drop_features

# lets drop the features
df = df.drop(drop_features, axis=1)

df.head()

"""## **Label Encoding**

* Label encoding is a pivotal data preprocessing technique that transforms categorical data into a numerical format, allowing machine learning algorithms to effectively process and analyze such information. In this process, each unique category within a categorical feature is assigned a corresponding integer label. Label encoding is particularly useful for algorithms that require numerical inputs, as it converts textual or categorical information into a structured numerical representation.
"""

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Iterate over categorical columns and apply label encoding
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

df.head()

label_encoder.classes_

import time

# Import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#import for preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder, MinMaxScaler



# import methods for measuring accuracy, precision, recall etc
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.metrics import roc_curve, auc, roc_auc_score

# lets extract the dependent and independent features
X = df.drop('class', axis=1)
y = df['class']

"""## **Split the Data**

* Subsequently, the data set was divided into training and test sets in a ratio of 80:20. This common practice ensures that a substantial portion (80%) of the data is used to train the machine learning model, while the remaining portion (20%) is reserved for evaluating its performance. This distribution helps in assessing the ability of the model to generalize to unseen data. By following this widely recognized division, the study maintains a rigorous approach to model evaluation and validation, which contributes to the reliability of the results obtained.
"""

# lets split the dataset stratified
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# lets try to check the shape of training,testing
print("training shape :",X_train.shape)
print("testing shape :", X_test.shape)

"""## **Normalize the Data**

* After the dataset partition, the next step involved normalizing the data using the Min-Max scaling technique. Min-Max scaling transforms numerical features to a common range, typically between 0 and 1, by subtracting the minimum value and dividing by the range (maximum - minimum). This normalization process ensures that all features contribute equally to the model training, regardless of their original scales. By minimizing the impact of varying scales, the model becomes more robust and accurate in capturing patterns and making predictions. Min-Max scaling is a crucial preprocessing step that enhances the performance and convergence of machine learning algorithms.
"""

'''
Feature scaling marks the end of the data preprocessing in Machine Learning. It is a method to standardize the independent variables of a dataset within a specific range.
In other words, feature scaling limits the range of variables so that you can compare them on common grounds.

'''

_scaler = MinMaxScaler()

X_train_std = _scaler.fit_transform(X_train)
X_test_std = _scaler.transform(X_test)

"""## **Feature Selection chi2**

* After feature normalization, feature selection was performed using the SelectKBest method from the sklearn.feature_selection module along with the chi-squared (chi2) metric. The goal of this process was to identify the most important features for modeling. The importance scores of the elements were ranked in descending order to visualize their cumulative impact. A cumulative score plot was created demonstrating how the cumulative feature scores contribute to the overall importance.
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

features = SelectKBest(score_func=chi2, k=X_train_std.shape[1])

# fit features to the training dataset
fit = features.fit(X_train_std, y_train)

# Get the indices sorted by most important to least important
indices = np.argsort(fit.scores_)[::-1]

# To get feature names
features_ = []
for i in range(len(X_train.columns.tolist())):
    features_.append(X_train.columns[indices[i]])

# Now plot
plt.figure(figsize=(15,5))
plt.bar(features_, fit.scores_[indices], color='r', align='center')
plt.xticks(rotation=90)
plt.show()

A -> 0.23
B -> 0.15
0.23+0.15

# sort the features by importance score
feature_importances = zip(X_train.columns, features.scores_)
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]

x_values = list(range(len(feature_importances)))

# plot the cumulative scores
cumulative_importances = np.cumsum(sorted_importances)
plt.figure(figsize=(15, 5))
plt.plot(x_values, cumulative_importances)

# Draw line at 99% of importance retained
value99 = cumulative_importances[-1]*0.99

plt.hlines(y = value99, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
# plt.vlines(x = value99, ymin=0, max=len(sorted_importances), color = 'r', linestyles = 'dashed')

# plt.xticks(x_values, sorted_features, rotation = 'vertical', fontsize=5)
plt.yticks([], [])
plt.xlabel('Total Feature', fontsize=15)
plt.title('A Chart to Show Cumulative Feature Scores', fontsize=15)
# plt.figure(figsize=(500,200))
plt.tight_layout()
plt.savefig('cum_features_CHI2.png', dpi=300)
plt.show()

# Print the total selected features
selected_features_count = np.sum(cumulative_importances <= value99)
print("Total selected features:", selected_features_count)

# lets get the 40 features
features = SelectKBest(score_func=chi2, k=selected_features_count)
fit = features.fit(X_train_std, y_train)
X_train_std = fit.transform(X_train_std)
X_test_std = fit.transform(X_test_std)

# lets check the 40 features that we will used for train the model
new_features = X_train.columns[features.get_support(indices=True)]
new_features = pd.DataFrame(new_features.tolist(), columns=['Best Features'])
print(tabulate(new_features, headers = 'keys', tablefmt = 'psql'))

"""## **Model Building**

### **Performance Metrics**

Here's an explanation of each performance metric, along with their mathematical formulas including the TP (True Positives), FP (False Positives), TN (True Negatives), and FN (False Negatives):
1. Accuracy:
Accuracy is the ratio of correctly predicted instances (TP and TN) to the total number of instances in the dataset.

          Mathematical Formula:
          Accuracy = (TP + TN) / (TP + TN + FP + FN)

2. Precision:
Precision is the ratio of correctly predicted positive instances (TP) to the total instances predicted as positive (TP and FP).

        Mathematical Formula:
        Precision = TP / (TP + FP)

3. Recall (Sensitivity or True Positive Rate):
Recall is the ratio of correctly predicted positive instances (TP) to the total actual positive instances (TP and FN).

        Mathematical Formula:
        Recall = TP / (TP + FN)

4. F1 Score:
The F1 score is the harmonic mean of precision and recall, emphasizing the balance between them.

        Mathematical Formula:
        F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

5. Classification Report:
A classification report provides a comprehensive overview of various performance metrics including TP, FP, TN, FN, precision, recall, and F1 score for each class.

6. Confusion Matrix:
A confusion matrix is a table that outlines the counts of TP, FP, TN, and FN for each class, facilitating a more detailed understanding of the model's performance.
These metrics, along with TP, FP, TN, and FN, collectively provide insights into the model's accuracy, precision, recall, and F1 score for different classes in classification tasks.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

Classes = label_encoder.classes_

0.9555

def results(model_name, y_pred, y_test, y_train,y_pred_train):
    target_names = ["{}".format(Classes[i]) for i in range(len(Classes))] # Define target names for classification report
    accuracy = round(accuracy_score(y_pred, y_test)*100,4)
    train_accuracy = round(accuracy_score(y_pred_train, y_train)*100,4)

    precision = round(precision_score(y_pred, y_test, average='macro')*100,4)
    recall = round(recall_score(y_pred, y_test, average='macro')*100,4)
    f1_scr = round(f1_score(y_pred, y_test, average='macro')*100,4)


    print("\nTraining Accuracy: {}%".format(train_accuracy))
    print("Testing Accuracy: {}%".format(accuracy))

    print("Precision: {}%".format(precision))
    print("Recall: {}%".format(recall))
    print("F1-Score: {}%".format(f1_scr))
    print()
    print("Classification Report:")
    print(classification_report(y_pred, y_test, target_names=target_names))
    print()
    print("Confusion Matrix:")
    fig, ax = plt.subplots(figsize=(7,5))
    ConfusionMatrixDisplay.from_predictions(y_pred, y_test,
                                            ax=ax,
                                            display_labels=target_names,
                                            xticks_rotation='vertical')
    plt.show()

    return {
        'Model':model_name,
        'Training Accuracy': train_accuracy,
        'Testing Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1_scr
    }

# Train KNN Model
knn = KNeighborsClassifier()
knn.fit(X_train_std, y_train)

# predict the model
y_pred = knn.predict(X_test_std)
y_pred_train = knn.predict(X_train_std)

# check the performance metrics
res = results(knn, y_pred, y_test, y_train, y_pred_train)

pd.DataFrame([res])

"""* Upon selecting the pertinent features and evaluating their performance using various metrics, the K-Nearest Neighbors (KNN) classification model was employed. KNN is a non-parametric algorithm that categorizes instances by considering the class of their nearest neighbors in the feature space. This intuitive approach allows the model to make predictions based on the majority class among its nearest neighbors. By leveraging the distance metric and adjusting the 'k' value (number of neighbors), the KNN model demonstrates its adaptability to different datasets and classification scenarios.

* The K-Nearest Neighbors (KNN) model exhibits remarkable predictive capabilities, particularly evident in the minimal cases of misclassification, specifically in the TCP and UDP flood categories.
"""
