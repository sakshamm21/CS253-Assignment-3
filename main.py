import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# Function to convert 'Crore+' and 'Lac+' to numeric values
def convert_to_numeric(value):
    value = str(value)
    if 'Crore+' in value:
        return float(value.replace('Crore+', '')) * 10**7
    elif 'Lac+' in value:
        return float(value.replace('Lac+', '')) * 10**5
    else:
        return pd.to_numeric(value, errors='coerce')

# Load the dataset
train_df = pd.read_csv('C:\\Users\\Saksham\\Desktop\\CS253\\ass3\\train.csv')
test_df = pd.read_csv('C:\\Users\\Saksham\\Desktop\\CS253\\ass3\\test.csv')

# Preprocessing
# Convert 'Total Assets' and 'Liabilities' from string to numeric
train_df['Total Assets'] = train_df['Total Assets'].apply(lambda x: convert_to_numeric(x))
train_df['Liabilities'] = train_df['Liabilities'].apply(lambda x: convert_to_numeric(x))
test_df['Total Assets'] = test_df['Total Assets'].apply(lambda x: convert_to_numeric(x))
test_df['Liabilities'] = test_df['Liabilities'].apply(lambda x: convert_to_numeric(x))

# Fill missing values
train_df.fillna({'Criminal Case': 0, 'Total Assets': train_df['Total Assets'].median(), 'Liabilities': train_df['Liabilities'].median()}, inplace=True)
test_df.fillna({'Criminal Case': 0, 'Total Assets': test_df['Total Assets'].median(), 'Liabilities': test_df['Liabilities'].median()}, inplace=True)

# Encoding categorical features and the target variable
le = LabelEncoder()
train_df['Party'] = le.fit_transform(train_df['Party'])
test_df['Party'] = le.transform(test_df['Party'])

# 'Education' is target variable
train_df['Education'] = le.fit_transform(train_df['Education'])

# Features/Labels split
X = train_df[['Party', 'Criminal Case', 'Total Assets', 'Liabilities']]
y = train_df['Education']

X1 = test_df[['Party', 'Criminal Case', 'Total Assets', 'Liabilities']]

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X, y)

best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

# Model training with best parameters
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X, y)

# Predicting the Test set results
y_pred = model.predict(X1)

# Prepare the submission
test_predictions = model.predict(test_df[['Party', 'Criminal Case', 'Total Assets', 'Liabilities']])
test_df['Education'] = le.inverse_transform(test_predictions)  # Convert back to original labels if needed
test_df[['ID', 'Education']].to_csv('submission.csv', index=False)
