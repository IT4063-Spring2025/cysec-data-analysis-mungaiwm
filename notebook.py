#!/usr/bin/env python
# coding: utf-8

# ## In-Class Activity - Cyber Security Data Analysis 
# This notebook will guide you through the process of analyzing a cyber security dataset. Follow the TODO tasks to complete the assignment.
# 

# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_rows = 999
from notebook import df, dfDummies, dfNormalized, X, y, models


# # Step 2: Loading the dataset
# 
# TODO: Load the given dataset.

# In[107]:


link = "./Data/CySecData.csv"


# # Step 3: Display the first few rows of the dataset
# TODO: Import the necessary libraries for data analysis and visualization.

# In[108]:


# display the first 5 rows of the dataset
def display_data():
    df = pd.read_csv(link)
    print(df.head(5))
    return df
df = display_data()


# # Step 4: Initial info on the dataset.
# 
# TODO: Provide a summary of the dataset.

# In[109]:


# provide a summary of the dataset
def summary_data():
    print(df.info())
    print(df.describe())
    return df
df = summary_data()


# # Step 5: Creating dummy variables
# TODO: Create dummy variables for the categorical columns except for the label column "class".

# In[110]:


# Create dummy variables for categorical columns except for the label column "class"
def create_dummies(df):
    # Select categorical columns excluding 'class'
    categorical_columns = [col for col in df.select_dtypes(include=['object']).columns if col != 'class']

    # Create dummy variables
    dfDummies = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    return dfDummies

# Assuming 'df' is the original DataFrame
dfDummies = create_dummies(df)

# Display the first few rows of the transformed DataFrame
print(dfDummies.head())



# # Step 6: Dropping the target column
# TODO: Drop the target column 'class' from the dataset.

# In[111]:


# Drop the target column 'class' from the dataset
def drop_target(df):
    if 'class' in df.columns:
        df = df.drop(columns=['class'])
    return df


# In[112]:


#Drop the target column 'class' from the dataset.
def drop_target():
    df.drop(columns=["class"], inplace=True)
    return df
df = drop_target()


# In[113]:


# add a column 'class' to the dataset
def add_class_column(df):
    df['class'] = df['class'].map({'no': 0, 'yes': 1})
    return df


# # Step 7: Importing the Standard Scaler
# TODO: Import the `StandardScaler` from `sklearn.preprocessing`.

# In[114]:


# import the StandardScaler from sklearn.preprocessing.
from sklearn.preprocessing import StandardScaler
# Create an instance of the StandardScaler.
scaler = StandardScaler()


# # Step 8: Scaling the dataset
# TODO: Scale the dataset using the `StandardScaler`.

# In[115]:


# Check for non-numeric columns
print(dfDummies.dtypes)


# In[116]:


#  drop non-numeric columns
dfDummies_numeric = dfDummies.select_dtypes(include=['number'])


# In[117]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def scale_data(df):
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

dfNormalized = scale_data(dfDummies_numeric)
print(dfNormalized.head())


# # Step 9: Splitting the dataset
# TODO: Split the dataset into features (X) and target (y).

# In[118]:


def split_dataset(df, target_column='class'):
    # Features (X): All columns except the target column
    X = df.loc[:, df.columns != target_column]
    # Target (y): The target column values
    y = df[target_column]
    return X, y




# # Step 10: Importing the required libraries for the models
# TODO: Import the necessary libraries for model training and evaluation.

# In[119]:


# import the necessary libraries for model training and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# # Step 11: Defining the models (Logistic Regression, Support Vector Machine, Random Forest)
# TODO: Define the models to be evaluated.

# In[120]:


# Define the models to be evaluated
models = []
models.append(('LR', LogisticRegression()))
models.append(('SVM', SVC()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
from sklearn.model_selection import cross_val_score




# # Step 12: Evaluating the models
# TODO: Evaluate the models using 10 fold cross-validation and display the mean and standard deviation of the accuracy.
# Hint: Use Kfold cross validation and a loop

# In[121]:


# Evaluate the models using 10 fold cross-validation and display the mean and standard deviation of the accuracy.

def evaluate_models(X, y, models):
    results = []
    names = []
    for name, model in models:
        cv_results = cross_val_score(model, X, y, cv=10, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    return results, names


# In[124]:


# Ensure all columns are numeric before scaling
def scale_data(df):
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    scaled_data = scaler.fit_transform(numeric_df)    # Scale the numeric data
    df_scaled = pd.DataFrame(scaled_data, columns=numeric_df.columns)  # Convert back to DataFrame
    return df_scaled

# Assuming 'dfDummies' is the DataFrame to be scaled
dfNormalized = scale_data(dfDummies)

# Display the first few rows of the scaled DataFrame
print(dfNormalized.head())


# # Step 13: Converting the notebook to a script
# TODO: Convert the notebook to a script using the `nbconvert` command.

# In[122]:


# Convert the notebook to a script using the `nbconvert` command.

get_ipython().system('jupyter nbconvert --to script notebook.ipynb')


# In[123]:


# Verify the script conversion
get_ipython().system('ls')

