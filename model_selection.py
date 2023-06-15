import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

# Load the preprocessed dataset
df = pd.read_csv('Data/Clean_Data.csv')

y = df['min_price']
X = df[['total_postings', 'total_tickets', 'face_value', 'days_to_show', 'num_blogs',
        'num_news', 'num_reviews', 'discovery', 'familiarity', 'hotness', 'num_years_active']]

# Reshape the data
X = X.values.reshape(-1, 11)
y = y.values.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit multiple linear regression model to the training data
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Print model summary
print('Multiple Linear Regression Model')
print('=================================')
print('Train score:', model_lr.score(X_train, y_train))
print('Test score:', model_lr.score(X_test, y_test))
print('Coefficients:', model_lr.coef_[0])
print('Intercept:', model_lr.intercept_)

# Fit ridge regression model to the training data
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)

# Print model summary
print('\nRidge Regression Model')
print('======================')
print('Train score:', model_ridge.score(X_train, y_train))
print('Test score:', model_ridge.score(X_test, y_test))
print('Coefficients:', model_ridge.coef_[0])
print('Intercept:', model_ridge.intercept_)

# Fit Lasso regression model to the training data
model_lasso = Lasso(alpha=0.1)
model_lasso.fit(X_train, y_train)

# Print model summary
print('\nLasso Regression Model')
print('======================')
print('Train score:', model_lasso.score(X_train, y_train))
print('Test score:', model_lasso.score(X_test, y_test))
print('Coefficients:', model_lasso.coef_)
print('Intercept:', model_lasso.intercept_)

# Calculate mean absolute error for multiple linear regression model
y_pred_lr = model_lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print('\nMultiple Linear Regression Model')
print('=================================')
print('Mean absolute error:', mae_lr)

# Calculate mean absolute error for ridge regression model
y_pred_ridge = model_ridge.predict(X_test)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
print('\nRidge Regression Model')
print('======================')
print('Mean absolute error:', mae_ridge)

# Calculate mean absolute error for Lasso regression model
y_pred_lasso = model_lasso.predict(X_test)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
print('\nLasso Regression Model')
print('======================')
print('Mean absolute error:', mae_lasso)