#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><b>Assignment 4: Pipelines and Hyperparameter Tuning</b></font>
# 
# ***
# * **Full Name** = David Rodriguez
# * **UCID** = 30145288
# ***

# <font color='Blue'>
# In this assignment, you will be putting together everything you have learned so far. You will need to find your own dataset, do all the appropriate preprocessing, test different supervised learning models, and evaluate the results. More details for each step can be found below. You will also be asked to describe the process by which you came up with the code. More details can be found below. Please cite any websites or AI tools that you used to help you with this assignment.
# </font>

# <font color='Red'>
# For this assignment, in addition to your .ipynb file, please also attach a PDF file. To generate this PDF file, you can use the print function (located under the "File" within Jupyter Notebook). Name this file ENGG444_Assignment##__yourUCID.pdf (this name is similar to your main .ipynb file). We will evaluate your assignment based on the two files and you need to provide both.
# </font>
# 
# 
# |         **Question**         | **Point(s)** |
# |:----------------------------:|:------------:|
# |  **1. Preprocessing Tasks**  |              |
# |              1.1             |       2      |
# |              1.2             |       2      |
# |              1.3             |       4      |
# | **2. Pipeline and Modeling** |              |
# |              2.1             |       3      |
# |              2.2             |       6      |
# |              2.3             |       5      |
# |              2.4             |       3      |
# |     **3. Bonus Question**    |     **2**    |
# |           **Total**          |    **25**    |

# ## **0. Dataset**
# 
# This data is a subset of the **Heart Disease Dataset**, which contains information about patients with possible coronary artery disease. The data has **14 attributes** and **294 instances**. The attributes include demographic, clinical, and laboratory features, such as age, sex, chest pain type, blood pressure, cholesterol, and electrocardiogram results. The last attribute is the **diagnosis of heart disease**, which is a categorical variable with values from 0 (no presence) to 4 (high presence). The data can be used for **classification** tasks, such as predicting the presence or absence of heart disease based on the other attributes.

# In[21]:


import pandas as pd

# Define the data source link
_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'

# Read the CSV file into a Pandas DataFrame, considering '?' as missing values
df = pd.read_csv(_link, na_values='?',
                 names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                        'ca', 'thal', 'num'])

# Display the DataFrame
display(df)


# # **1. Preprocessing Tasks**
# 
# - **1.1** Find out which columns have more than 60% of their values missing and drop them from the data frame. Explain why this is a reasonable way to handle these columns. **(2 Points)**
# 
# - **1.2** For the remaining columns that have some missing values, choose an appropriate imputation method to fill them in. You can use the `SimpleImputer` class from `sklearn.impute` or any other method you prefer. Explain why you chose this method and how it affects the data. **(2 Points)**
# 
# - **1.3** Assign the `num` column to the variable `y` and the rest of the columns to the variable `X`. The `num` column indicates the presence or absence of heart disease based on the angiographic disease status of the patients. Create a `ColumnTransformer` object that applies different preprocessing steps to different subsets of features. Use `StandardScaler` for the numerical features, `OneHotEncoder` for the categorical features, and `passthrough` for the binary features. List the names of the features that belong to each group and explain why they need different transformations. You will use this `ColumnTransformer` in a pipeline in the next question. **(4 Points)**

# <font color='Green'><b>Answer:</b></font>
# 
# - **1.1** .....................
# ## Find out which columns have more than 60% of their values missing and drop them from the data frame. Explain why this is a reasonable way to handle these columns.
# As shown below the columns with more than 60% of its entries empty or NaN, are slope, ca, and thal. The reason this is a responsible way to handle these columns is because filling them in leads to high bias. Most metrics such as using the mean, most_frequent, and a constant are not suitable when it substitutes a significant portion of the data. Additionally, since we still have several other features that can be utilized to train and predict our model it is not irresponsible to drop the column. 
# 
# Upon further examination the columns being dropped slope refers to the slop of the peak exercise ST segment on the patient's ECG during testing. 'ca' is the may stand for the number of major vessels colored by fluoroscopy or the coronary arteries. 'thal' refers to thalassemia, which is a genetic blood disorder that affects the production of hemoglobin.  It is characterized by abnormal hemoglobin production. These are likely to be empty because they are harder to examine and are not often recorded. In future models, including them may be ideal.
# 
# 

# In[22]:


# 1.1
# Add necessary code here.
# print(len(df))
# print(df.isnull().sum())
missing_values = df.isnull().sum() # Count the number of missing values in each column
missing_values = missing_values[missing_values > 0.6 * len(df)] # Select columns with more than 60% missing values
print(missing_values) # Display the columns with more than 60% missing values
df = df.drop(missing_values.index, axis=1) # Drop the columns with more than 60% missing values
print(df.isnull().sum()) # Display the number of missing values in each column

# Inspect the nature of the data in each column to see if its binary, numerical, or categorical
# print('trestbps:', df['trestbps'].unique())
# print('chol:', df['chol'].unique())
# print('fbs:', df['fbs'].unique())
# print('restecg:', df['restecg'].unique())
# print('thalach:', df['thalach'].unique())
# print('exang:', df['exang'].unique())


# <font color='Green'><b>Answer:</b></font>
# 
# - **1.2** ..................... 
# ## For the remaining columns that have some missing values, choose an appropriate imputation method to fill them in. You can use the `SimpleImputer` class from `sklearn.impute` or any other method you prefer. Explain why you chose this method and how it affects the data.
# 
# There are multiple ways of filling in null values, utilizing the feature's mean, the feature's most_frequent, and a constant. Other methods in basic models can use ffil, bfill and different interpolation methods, these are not able to be applied to our model due to the unorganized data, and high dimension disabling this possibility. The mean, most_frequent, and constant are good options here depending on the data type. The mean way is valid for numerical and continous data since it still somewhat preserves the nature of the data. This will likely be an outlier in the model however, when data is scarce including it could prove beneficial. For categorical and binary data types, it is best to not use this method since it is possible to create a whole new unique data type which can undermine the function of the model. That's why I utilized most_frequent for these data types. In the examples: the binary and categorical features that had empty values were the, fbs, restecg, exang, this was evident upon inspecting the unique data types, and reading the description online. The numerical feature were chol, trestbps and thalch due to them being continous data types.

# In[23]:


# 1.2
# Add necessary code here.
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# for x in df[df.isnull().sum().index]:
#     print(x)
#     print(df[x].unique())

Binary_And_Categorical = ['fbs','restecg','exang']
Numerical_Features = ['chol','trestbps','thalach']
imputerBinaryAndCategorical = SimpleImputer(strategy='most_frequent') # Create an imputer object with a mean filling strategy
imputerNumerical = SimpleImputer(strategy='mean') # Create an imputer object with a mean filling strategy

preprocessor = ColumnTransformer(
    transformers=[
        ('num', imputerNumerical, Numerical_Features),  # Impute numerical features with mean
        ('cat', imputerBinaryAndCategorical, Binary_And_Categorical)  # Impute categorical features with most frequent
    ],
    remainder='passthrough'  # Include all remaining columns in the output DataFrame
)

other_columns = list(set(df.columns) - set(Numerical_Features) - set(Binary_And_Categorical))
df_imputed = pd.DataFrame(preprocessor.fit_transform(df), columns=Numerical_Features + Binary_And_Categorical + other_columns)

# print(df_imputed) # Display the number of missing values in each column
# print(df_imputed.isnull().sum()) # Display the number of missing values in each column

print(df_imputed.isnull().sum()) # Display the number of missing values in each column
# for x in df.columns:
#     # print(df[x], df[x].dtype)


# <font color='Green'><b>Answer:</b></font>
# 
# - **1.3** .....................
# ## Assign the `num` column to the variable `y` and the rest of the columns to the variable `X`. The `num` column indicates the presence or absence of heart disease based on the angiographic disease status of the patients. Create a `ColumnTransformer` object that applies different preprocessing steps to different subsets of features. Use `StandardScaler` for the numerical features, `OneHotEncoder` for the categorical features, and `passthrough` for the binary features. List the names of the features that belong to each group and explain why they need different transformations. You will use this `ColumnTransformer` in a pipeline in the next question.
# 

# In[28]:


# 1.3
# Add necessary code here.

# Assign the `num` column to the variable `y` and the rest of the columns to the variable `X`. The `num` column indicates the presence or absence of heart disease based on the angiographic disease status of the patients.
# Create a `ColumnTransformer` object that applies different preprocessing steps to different subsets of features. 
#Use `StandardScaler` for the numerical features, `OneHotEncoder` for the categorical features, and `passthrough` for the binary features. List the names of the features that belong to each group and explain why they need different transformations. You will use this `ColumnTransformer` in a pipeline in the next question.
y = df_imputed['num']
X = df_imputed.drop(columns=['num'])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

print(X.dtypes)
# num2 =list(df.select_dtypes(include=['float64']).columns)
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['cp','restecg',]
binary_features = ['sex', 'fbs', 'exang']



preprocessor = ColumnTransformer([
            ('Numerical', StandardScaler(), numerical_features),
            ('Categorical', OneHotEncoder(sparse_output = False), categorical_features),
            ('Binary', 'passthrough', binary_features)
    ]
)



# # **2. Pipeline and Modeling**
# 
# - **2.1** Create **three** `Pipeline` objects that take the column transformer from the previous question as the first step and add one or more models as the subsequent steps. You can use any models from `sklearn` or other libraries that are suitable for binary classification. For each pipeline, explain **why** you selected the model(s) and what are their **strengths and weaknesses** for this data set. **(3 Points)**
# 
# - **2.2** Use `GridSearchCV` to perform a grid search over the hyperparameters of each pipeline and find the best combination that maximizes the cross-validation score. Report the best parameters and the best score for each pipeline. Then, update the hyperparameters of each pipeline using the best parameters from the grid search. **(6 Points)**
# 
# - **2.3** Form a stacking classifier that uses the three pipelines from the previous question as the base estimators and a meta-model as the `final_estimator`. You can choose any model for the meta-model that is suitable for binary classification. Explain **why** you chose the meta-model and how it combines the predictions of the base estimators. Then, use `StratifiedKFold` to perform a cross-validation on the stacking classifier and present the accuracy scores and F1 scores for each fold. Report the mean and the standard deviation of each score in the format of `mean ± std`. For example, `0.85 ± 0.05`. Interpret the results and compare them with the baseline scores from the previous assignment. **(5 Points)**
# 
# - **2.4**: Interpret the final results of the stacking classifier and compare its performance with the individual models. Explain how stacking classifier has improved or deteriorated the prediction accuracy and F1 score, and what are the possible reasons for that. **(3 Points)**

# <font color='Green'><b>Answer:</b></font>
# 
# - **2.1** .....................

# In[25]:


# 2.1
# Add necessary code here.
# create three pipeline objects that take the column transformer from the pervious question as the frist step and add one or more models as the subsequeent steps. You can use any models from sklearn or other libraries.
#or other libraries that are suitable for binary classification. For each pipeline, expalin why you selected the model(s) and what are tehir strengths and weakensses for thsi data set.add
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

LogisticRegression_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('LogisticRegression', LogisticRegression())
])
SVC_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('SVC', SVC())
])
GradientBoosting_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('GradientBoosting', GradientBoostingClassifier())
])



    


# <font color='Green'><b>Answer:</b></font>
# 
# - **2.2** .....................

# In[26]:


# 2.2
# Add necessary code here.
#  Use `GridSearchCV` to perform a grid search over the hyperparameters of each pipeline and 
# find the best combination that maximizes the cross-validation score.
# Report the best parameters and the best score for each pipeline. 
# Then, update the hyperparameters of each pipeline
# using the best parameters from the grid search.

from sklearn.model_selection import GridSearchCV, train_test_split
param_grid_LogisticRegression = {
    'LogisticRegression__C': [0.1, 1, 10, 100],
    'LogisticRegression__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
param_grid_SVC = {
    'SVC__C': [0.1, 1, 10, 100],
    'SVC__gamma': [1, 0.1, 0.01, 0.001],
    'SVC__kernel': ['rbf', 'poly', 'sigmoid','linear']
}
param_grd_GradientBoosting = {
    'GradientBoosting__n_estimators': [100, 200, 300],
    'GradientBoosting__learning_rate': [0.1, 0.01, 0.001],
    'GradientBoosting__max_depth': [3, 4, 5]
}
print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)
grid_LogisticRegression = GridSearchCV(estimator=LogisticRegression_pipeline,param_grid = param_grid_LogisticRegression,scoring='accuracy', cv=5)
grid_LogisticRegression.fit(X_train, y_train)

grid_SVC = GridSearchCV(SVC_pipeline, param_grid_SVC, cv=5)
grid_SVC.fit(X_train, y_train)

grid_GradientBoosting = GridSearchCV(estimator=GradientBoosting_pipeline, param_grid=param_grd_GradientBoosting,scoring='accuracy', cv=5)
grid_GradientBoosting.fit(X_train, y_train)

print("-----------------------------------")
print("Logistic Regression")
print("Best parameters for LogisticRegression: ", grid_LogisticRegression.best_params_)
print("Best score for LogisticRegression on training data: ", grid_LogisticRegression.best_score_)
print("Best accuracy score on testing data: ", grid_LogisticRegression.score(X_test, y_test))
print("-----------------------------------")
print("SVC")
print("Best parameters for SVC: ", grid_SVC.best_params_)
print("Best score for SVC on training on training data: ", grid_SVC.best_score_)
print("Best accuracy score on testing data: ", grid_SVC.score(X_test, y_test))

print("-----------------------------------")
print("GradientBoostingClassifier")
print("Best parameters for GradientBoosting: ", grid_GradientBoosting.best_params_)
print("Best score for GradientBoosting on training data: ", grid_GradientBoosting.best_score_)
print("Best accuracy score on testing data: ", grid_GradientBoosting.score(X_test, y_test))

LogisticRegression_pipeline.set_params(**grid_LogisticRegression.best_params_)
SVC_pipeline.set_params(**grid_SVC.best_params_)
GradientBoosting_pipeline.set_params(**grid_GradientBoosting.best_params_)



# <font color='Green'><b>Answer:</b></font>
# 
# - **2.3** .....................

# In[ ]:


# 2.3
# Add necessary code here.


# <font color='Green'><b>Answer:</b></font>
# 
# - **2.4** .....................

# **Bonus Question**: The stacking classifier has achieved a high accuracy and F1 score, but there may be still room for improvement. Suggest **two** possible ways to improve the modeling using the stacking classifier, and explain **how** and **why** they could improve the performance. **(2 points)**

# <font color='Green'><b>Answer:</b></font>
