import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
import matplotlib.pyplot as plt

# Load the preprocessed dataset
df = pd.read_csv('Data/Clean_Data.csv')

encoder = LabelEncoder()
city = encoder.fit_transform(df['city'].values)
city = np.array([city]).T
enc = OneHotEncoder()
a = enc.fit_transform(city)
a = a.toarray()
df = pd.concat([df, pd.DataFrame(a)], axis=1)
for i in range(len(np.unique(df['city'].values))):
    df = df.rename(columns={i: f'city_{i}'})

encoder = LabelEncoder()
state = encoder.fit_transform(df['state'].values)
state = np.array([state]).T
enc = OneHotEncoder()
a = enc.fit_transform(state)
a = a.toarray()
df = pd.concat([df, pd.DataFrame(a)], axis=1)
for i in range(len(np.unique(df['state'].values))):
    df = df.rename(columns={i: f'state_{i}'})

y = df['total_tickets']
x = []
for i in range(len(np.unique(df['city'].values))):
    x.append(f'city_{i}')
for i in range(len(np.unique(df['state'].values))):
    x.append(f'state_{i}')
X = df[x]
import pdb
pdb.set_trace()

# Reshape the data
X = X.values.reshape(-1, 30)
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
