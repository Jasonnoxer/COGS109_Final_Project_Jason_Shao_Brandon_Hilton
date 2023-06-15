import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
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

# Calculate mean absolute error for Lasso regression model
y_pred_lasso = model_lasso.predict(X_test)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
print('\nLasso Regression Model')
print('======================')
print('Mean absolute error:', mae_lasso)

# Create a scatter plot of predicted vs. actual values for the Lasso regression model
plt.scatter(y_test, y_pred_lasso)
plt.plot([0, 300], [0, 300], color='r', linestyle='--')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Scatter plot of predicted vs. actual values for Lasso regression model')
plt.legend(['Predicted vs Actual', 'Perfect alignment'])
plt.show()