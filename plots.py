import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Preprocessing
# Convert 'Total Assets' and 'Liabilities' from string to numeric
train_df['Total Assets'] = train_df['Total Assets'].apply(lambda x: convert_to_numeric(x))
train_df['Liabilities'] = train_df['Liabilities'].apply(lambda x: convert_to_numeric(x))

# Fill missing values
train_df.fillna({'Criminal Case': 0, 'Total Assets': train_df['Total Assets'].median(), 'Liabilities': train_df['Liabilities'].median()}, inplace=True)

# Plotting Distribution of Criminal Cases by Party
criminal_cases_by_party = train_df.groupby('Party')['Criminal Case'].sum().sort_values(ascending=False)
criminal_cases_by_party_percent = (criminal_cases_by_party / criminal_cases_by_party.sum()) * 100
plt.figure(figsize=(10, 6))
sns.barplot(x=criminal_cases_by_party_percent.index, y=criminal_cases_by_party_percent.values, palette='viridis')
plt.title('Percentage Distribution of Criminal Cases by Party')
plt.xlabel('Party')
plt.ylabel('Percentage of Criminal Cases')
plt.xticks(rotation=45)
plt.show()

# Plotting Distribution of Wealth by Party
wealth_by_party = train_df.groupby('Party')['Total Assets'].sum().sort_values(ascending=False)
wealth_by_party_percent = (wealth_by_party / wealth_by_party.sum()) * 100
plt.figure(figsize=(10, 6))
sns.barplot(x=wealth_by_party_percent.index, y=wealth_by_party_percent.values, palette='coolwarm')
plt.title('Percentage Distribution of Wealth by Party')
plt.xlabel('Party')
plt.ylabel('Percentage of Total Assets')
plt.xticks(rotation=45)
plt.show()

# Bonus Plot: Total Assets vs Liabilities
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total Assets', y='Liabilities', data=train_df, color='blue', alpha=0.6)
plt.title('Relationship Between Total Assets and Liabilities')
plt.xlabel('Total Assets')
plt.ylabel('Liabilities')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()
