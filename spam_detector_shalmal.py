#!/usr/bin/env python
# coding: utf-8

# # Spam Detector

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ## Retrieve the Data
# 
# The data is located at [https://static.bc-edx.com/mbc/ai/m4/datasets/spam-data.csv](https://static.bc-edx.com/mbc/ai/m4/datasets/spam-data.csv)
# 
# Dataset Source: [UCI Machine Learning Library](https://archive-beta.ics.uci.edu/dataset/94/spambase)
# 
# Import the data using Pandas. Display the resulting DataFrame to confirm the import was successful.

# In[3]:


# Import the data
data = pd.read_csv("https://static.bc-edx.com/mbc/ai/m4/datasets/spam-data.csv")
data.head()


# ## Predict Model Performance
# 
# You will be creating and comparing two models on this data: a Logistic Regression, and a Random Forests Classifier. Before you create, fit, and score the models, make a prediction as to which model you think will perform better. You do not need to be correct! 
# 
# Write down your prediction in the designated cells in your Jupyter Notebook, and provide justification for your educated guess.

# *As Random forest comprises multiple smaller and simpler decision trees that are each trained on a subset of the training data and predict a specific class. RF is predicted to perform better*

# ## Split the Data into Training and Testing Sets

# In[9]:


# Create the labels set `y` and features DataFrame `X`
X = data.copy()
X.drop("spam",axis=1,inplace=True)
#display data with spam coplumn dropped
X.head()


# In[59]:


#define target y
y = data["spam"].ravel()
y[:5]

# Check the balance of the labels variable (`y`) by using the `value_counts` function.
data["spam"].value_counts()


# In[29]:


# Split the data into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)


# ## Scale the Features

# Use the `StandardScaler` to scale the features data. Remember that only `X_train` and `X_test` DataFrames should be scaled.

# In[30]:


from sklearn.preprocessing import StandardScaler

# Create the StandardScaler instance
scaler = StandardScaler()


# In[31]:


# Fit the Standard Scaler with the training data
X_scaler = scaler.fit(X_train)


# In[32]:


# Scale the training data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# ## Create and Fit a Logistic Regression Model
# 
# Create a Logistic Regression model, fit it to the training data, make predictions with the testing data, and print the model's accuracy score. You may choose any starting settings you like. 

# In[51]:


# Train a Logistic Regression model and print the model score
from sklearn.linear_model import LogisticRegression
#Create a `LogisticRegression` model
logistic_regression_model = LogisticRegression(random_state=1)
# Fit the model
logistic_regression_model.fit(X_train_scaled, y_train)
print(f"Training Data Score: {logistic_regression_model.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {logistic_regression_model.score(X_test_scaled, y_test)}")


# In[60]:


# Make and save testing predictions with the saved logistic regression model using the test data
test_predictions = logistic_regression_model.predict(X_test_scaled)
results_df = pd.DataFrame({
    "Testing Data Predictions": test_predictions, 
    "Testing Data Actual Targets": y_test})
# Review the predictions
results_df


# In[61]:


# Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.
accuracy_score(y_test, test_predictions)


# ## Create and Fit a Random Forest Classifier Model
# 
# Create a Random Forest Classifier model, fit it to the training data, make predictions with the testing data, and print the model's accuracy score. You may choose any starting settings you like. 

# In[62]:


# Train a Random Forest Classifier model and print the model score
from sklearn.ensemble import RandomForestClassifier
# Create the random forest classifier instance
rf_model = RandomForestClassifier(n_estimators=128, random_state=1)
# Fit the model
rf_model = rf_model.fit(X_train_scaled, y_train)



# In[55]:


# Make and save testing predictions with the saved logistic regression model using the test data
predictions = rf_model.predict(X_test_scaled)

# Review the predictions
rf_results_df = pd.DataFrame({
    "Testing RF Data Predictions": predictions, 
    "Testing RF Data Actual Targets": y_test})
# Review the predictions
rf_results_df


# In[56]:


# Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.
acc_score = accuracy_score(y_test, predictions)
print(f"Accuracy Score : {acc_score}")


# ## Evaluate the Models
# 
# Which model performed better? How does that compare to your prediction? Write down your results and thoughts in the following markdown cell.

# *Replace the text in this markdown cell with your answers to these questions.*

# In[57]:


*Random Forest Classifier Model performed better as predicted, it shows 95.5% Accuracy Score vs the 92.7% accuracy of Logistic Regression Model*


# In[ ]:




