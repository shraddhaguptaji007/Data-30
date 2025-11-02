import pandas as pd

df = pd.read_csv('titanic.csv')

print(df.head())

print(df.info())
print(df.describe())

print(df.isnull().sum())

df['age'].fillna(df['age'].mean(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

print(df['sex'].value_counts())
print(df['embarked'].value_counts())
