# Loan Defaulters Prediction

## Project Overview
The **Loan Defaulters Prediction** project aims to develop a machine learning model to predict whether a borrower will default on a loan. By analyzing various features such as income, interest rates, and demographic information, the model can identify potential defaulters, helping financial institutions mitigate risk and make informed lending decisions.

## Scope and Objective
- **Scope**: This project involves data preprocessing, feature engineering, model training, evaluation, and validation to accurately predict loan defaults.
- **Objective**: To build a robust prediction model that can accurately classify borrowers into defaulters and non-defaulters, thereby helping financial institutions in making better lending decisions.

## Technology Stack
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: `Pandas`, `NumPy`
  - Data Visualization: `Matplotlib`, `Seaborn`
  - Machine Learning: `scikit-learn`
- **Dataset**: A CSV file containing loan-related data with features such as `income`, `rate_of_interest`, and `Status` (target variable).

## Data Preprocessing
### 1. Importing Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
We start by importing essential libraries for data manipulation, visualization, and model training.

### 2. Loading the Dataset
```python
df = pd.read_csv("Loan_Default.csv")
```
The dataset is loaded into a Pandas DataFrame for easier manipulation and analysis.

### 3. Initial Data Exploration
```python
df.head()
df.info()
df.isna().sum()
```
We check the structure of the dataset, including data types, missing values, and the first few rows of data.

### 4. Handling Missing Values and Data Cleaning
```python
df.drop(df[df['Gender'] == 'Sex Not Available'].index, inplace=True)
df.fillna({
    'income': df.income.median(),
    'rate_of_interest': df['rate_of_interest'].mode()[0],
    'Interest_rate_spread' : df['Interest_rate_spread'].mode()[0],
    'Upfront_charges': df['Upfront_charges'].mode()[0]
}, inplace=True)
df.dropna(inplace=True)
```
- Removed irrelevant data such as rows where `Gender` is `Sex Not Available`.
- Filled missing values using statistical methods like median and mode.
- Dropped any remaining rows with missing values to ensure a clean dataset.

### 5. Encoding Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X = df.drop("Status", axis=1)
y = df.Status
X.drop(['ID', "year"], axis=1, inplace=True)
for col in list(X.columns):
    X[col] = le.fit_transform(X[col])
```
Categorical variables are converted into numerical values using `LabelEncoder` to make the data compatible with machine learning algorithms.

### 6. Splitting the Dataset
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 23)
```
The dataset is split into training and testing sets with an 80-20 ratio to evaluate model performance on unseen data.

## Model Training and Evaluation
### 1. Training a Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
rfs = RandomForestClassifier(n_estimators = 50, max_depth = 5)
rfs.fit(X_train, y_train)
```
We use a `RandomForestClassifier`, which is an ensemble method known for its accuracy and robustness. The model is trained on the training set.

### 2. Evaluating Model Performance on Training Data
```python
train_pred = rfs.predict(X_train)
accuracy_score(y_train, train_pred)
f1_score(y_train, train_pred)
```
The model's performance is evaluated on the training set using metrics like accuracy and F1 score.

### 3. Confusion Matrix Visualization
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, train_pred)
sns.heatmap(cm, cmap='viridis', annot=True, fmt='.2f')
plt.show()
```
A confusion matrix is plotted to visualize the true positives, true negatives, false positives, and false negatives.

### 4. Evaluating Model Performance on Test Data
```python
test_pred = rfs.predict(X_test)
accuracy_score(y_test, test_pred)
f1_score(y_test, test_pred)
sns.heatmap(confusion_matrix(y_test, test_pred), cmap='viridis', annot=True, fmt='.2f')
plt.show()
print(classification_report(y_test, test_pred))
```
The model is then evaluated on the test set to determine how well it generalizes to unseen data. We calculate accuracy, F1 score, and plot the confusion matrix. A classification report is printed to give a detailed breakdown of precision, recall, and F1 scores for each class.

## Future Extensions
- **Hyperparameter Tuning**: Experiment with different hyperparameters (e.g., number of trees, max depth) to further improve model performance.
- **Feature Engineering**: Create new features that could potentially enhance the model’s predictive power, such as interaction terms or domain-specific indicators.
- **Model Comparison**: Test other machine learning models such as Gradient Boosting Machines (GBMs), XGBoost, or neural networks to compare performance.
- **Handling Imbalanced Data**: If the dataset is imbalanced (e.g., far more non-defaulters than defaulters), techniques like SMOTE (Synthetic Minority Over-sampling Technique) or weighted loss functions could be employed.
- **Deploying the Model**: Integrate the model into a web application or financial software to provide real-time predictions for loan applications.

## References
- [Pandas Documentation](https://pandas.pydata.org/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

## Contact

For any queries, feel free to reach out to me at [jameers2003@gmail.com](mailto:jameers2003@gmail.com).
