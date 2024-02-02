# DOS Detector ML-KNN

## Overview
DOS Detector ML-KNN is a Python script designed for detecting Denial-of-Service (DOS) attacks using the K-Nearest Neighbors (KNN) machine learning algorithm. The script reads and analyzes network traffic datasets related to different DOS attack types, including TCP, UDP, and HTTP floods. By employing feature engineering, data preprocessing, and KNN classification, the script aims to identify and classify DOS attacks in real-time.

## Usage
1. **Data Preparation:**
   - Ensure that your network traffic datasets are organized in a directory structure. The script provides functionality to locate and load CSV files within a specified directory.

   ```python
   # Example: Provide the directory path where you want to start searching for CSV files
   starting_directory = "/content/drive/MyDrive/BoT-IoT Dataset/DoS"
   csv_files = find_csv_files(starting_directory)
2. **Load and Explore Datasets:**

	- Load the relevant CSV files for TCP, UDP, and HTTP DOS attacks.
   - Explore the first 5 rows of each dataset to understand the data structure.
		``` python Copy code
			df_dos_TCP = pd.read_csv("/content/drive/MyDrive/BoT-IoT Dataset/DoS/DoS_TCP/DoS_TCP[1].csv")
			df_dos_UDP = pd.read_csv("/content/drive/MyDrive/BoT-IoT Dataset/DoS/DoS_UDP/DoS_UDP[1].csv")
			df_dos_HTTP = pd.read_csv("/content/drive/MyDrive/BoT-IoT Dataset/DoS/DoS_HTTP/DoS_HTTP[1].csv")
      df_dos_TCP.head()
      df_dos_UDP.head()
      df_dos_HTTP.head()
3. **Data Preprocessing:**
  - Filter and preprocess the datasets, removing irrelevant features and standardizing class labels. Explore the general information and handle any missing values.
    ``` python Copy code
    # Example: Remove irrelevant features and standardize class labels
    df_dos_TCP_ = df_dos_TCP.drop(['Timestamp','Label', 'Cat', 'Flow_ID'], axis=1)
    df_dos_TCP_ = df_dos_TCP_.rename(columns={'Sub_Cat':'class'})
    df_dos_TCP_['class']  = df_dos_TCP_['class'].map({'DoS_TCP':'TCP_FLOOD'})
4. **Feature Selection:**
  - Utilize the chi-squared (chi2) metric to perform feature selection and choose the most important features for modeling.
    ``` python Copy code
    # Example: Perform feature selection using chi2 metric
    features = SelectKBest(score_func=chi2, k=X_train_std.shape[1])
    fit = features.fit(X_train_std, y_train)
5. **Model Training and Evaluation:**

  - Train the KNN model on the preprocessed and selected features. Evaluate the model's performance using accuracy, precision, recall, and F1-score.
    ``` python Copy code
    # Example: Train KNN Model and Evaluate Performance
    knn = KNeighborsClassifier()
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    y_pred_train = knn.predict(X_train_std)

# Example: Check the performance metrics
    res = results(knn, y_pred, y_test, y_train, y_pred_train)
# Requirements:

    Python 3.x
    Pandas
    NumPy
    Matplotlib
    Seaborn
    TQDM
    Tabulate
    Scikit-learn
# License
This tool is open-source and available under the MIT License.
