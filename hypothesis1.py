import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the preprocessed dataset
df = pd.read_csv('Data/Clean_Data.csv')

y = df['total_tickets']
X = df[['num_blogs', 'num_news', 'num_reviews', 'discovery', 'familiarity', 'hotness', 'num_years_active']]

# Reshape the data
X = X.values.reshape(-1, 7)
y = y.values.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# Fit a linear regression model to the training data
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train)
results = model.fit()

# Print model summary
print(results.summary())

y_pred = results.predict(sm.add_constant(X_test))
mae = np.mean(np.abs(y_pred - y_test))
print('Mean absolute error:', mae)

# Plot the data and linear regression line

plt.scatter(X_train[:, 1], y_train, color='blue', label='Training Data')
plt.scatter(X_test[:, 1], y_test, color='red', label='Testing Data')
plt.plot(X_train[:, 1], results.predict(X_train), color='black', linewidth=3, label='Linear Regression')
plt.xlabel('Num Blogs')
plt.ylabel('Total Tickets')
plt.legend()
plt.show()