# Imputers on Titanic Dataset

This project aims to predict survival on the Titanic using a machine learning pipeline. The dataset used is the well-known Titanic dataset from Kaggle. A key aspect of this project is the use of various imputation techniques to handle missing data. The workflow includes data loading, feature engineering, preprocessing with different imputers, and training a logistic regression model.

## Project Description

### Data Loading and Feature Engineering

1. **Data Loading**: The dataset is loaded using pandas, and initial exploration is done using the `head()` method.
2. **Feature Engineering**: A new feature `family` is created by summing `SibSp` (siblings/spouses aboard) and `Parch` (parents/children aboard). Irrelevant columns such as `Cabin`, `Name`, `PassengerId`, and `Ticket` are dropped to simplify the dataset.

### Data Splitting

The dataset is split into training and testing sets using an 80-20 split. This helps in evaluating the model's performance on unseen data.

### Data Preprocessing with Imputers

The preprocessing steps involve handling missing values and preparing the data for model training. Two main types of features are considered: numerical and categorical. Different imputers are used to handle missing values in these features.

1. **Numerical Features**:
    - **Imputation with KNNImputer**: Missing values in numerical features (`Age` and `Fare`) are imputed using KNNImputer. This method utilizes the k-nearest neighbors to estimate the missing values based on the similarity of the available data points. Specifically, `n_neighbors=5` and `weights='distance'` are used to give more influence to closer neighbors.
    - **Scaling with StandardScaler**: After imputation, numerical features are scaled to have zero mean and unit variance using StandardScaler.

2. **Categorical Features**:
    - **Imputation with SimpleImputer**: Missing values in categorical features (`Embarked` and `Sex`) are imputed using SimpleImputer. The strategy chosen is `most_frequent`, which replaces missing values with the most frequent value in each column. This approach ensures that categorical features are filled with the most common category, which is often a reasonable assumption for categorical data.
    - **Encoding with OneHotEncoder**: After imputation, categorical features are one-hot encoded to convert them into a format suitable for the logistic regression model.

### Model Training and Hyperparameter Tuning

A logistic regression model is used for classification. The pipeline for preprocessing and the classifier is combined, and GridSearchCV is used for hyperparameter tuning. The parameters tuned include:

- Number of neighbors (`n_neighbors`) for KNNImputer.
- Imputation strategy (`strategy`) for SimpleImputer.
- Regularization strength (`C`) for the logistic regression classifier.

### Evaluation

After finding the best hyperparameters using GridSearchCV, the model is retrained on the entire training set and evaluated on the test set. The accuracy score is computed to measure the model's performance.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/abhiram-k-2223/imputers.git
    cd imputer
    ```

2. **Install dependencies**:
    ```bash
    pip install pandas scikit-learn
    ```

3. **Run the Jupyter notebook or script**:
    - If using a Jupyter notebook, open `imputers.ipynb` and run the cells sequentially.
   

4. **Output**:
    - The best hyperparameters found by GridSearchCV.
    - The internal cross-validation score.
    - The test accuracy score.

This project demonstrates how to handle missing data using different imputation techniques and preprocessing methods, integrated into a machine learning pipeline to predict outcomes effectively.
