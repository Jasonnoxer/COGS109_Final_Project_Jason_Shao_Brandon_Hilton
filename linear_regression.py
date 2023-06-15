import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

# Load the preprocessed dataset
df = pd.read_csv('Data/Clean_Data.csv')
import pdb
pdb.set_trace()
y = df['min_price']

# Drop the some column
df = df.drop('date', axis=1)

# encoder = LabelEncoder()
# state = encoder.fit_transform(df['state'].values)
# state = np.array([state]).T
# enc = OneHotEncoder()
# a = enc.fit_transform(state)
# a = a.toarray()
# df = pd.concat([df, pd.DataFrame(a)], axis=1)
# for i in range(len(np.unique(df['state'].values))):
#     df = df.rename(columns={i: f'state_{i}'})

df = df.drop('artist', axis=1)
df = df.drop('city', axis=1)
df = df.drop('venue', axis=1)
df = df.drop('state', axis=1)
df = df.drop('ticket_vendor', axis=1)
df = df.drop('min_price', axis=1)
df = df.drop('max_price', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit a linear regression model to the training data
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(lr, X_train, y_train, cv=5)
print('Cross-validation scores:', cv_scores)
print('Mean cross-validation score:', np.mean(cv_scores))

# Make predictions on the testing data
y_pred = lr.predict(X_test)

# # Calculate the mean squared error of the model on the testing data
# mse = np.mean((y_pred - y_test) ** 2)
# print('Mean squared error:', mse)

# Calculate the mean absolute error of the model on the testing data
mae = np.mean(np.abs(y_pred - y_test))
print('Mean absolute error:', mae)
