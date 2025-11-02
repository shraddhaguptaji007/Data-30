import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats

df = sns.load_dataset('titanic')

print("🔹 First 5 Rows:")
print(df.head())

print("\n🔹 Basic Info:")
print(df.info())

print("\n🔹 Missing Values Before Cleaning:")
print(df.isnull().sum())

df['age'].fillna(df['age'].median(), inplace=True)

df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['deck'] = df['deck'].astype(str)
df['deck'].fillna('Unknown', inplace=True)


print("\n🔹 Missing Values After Cleaning:")
print(df.isnull().sum())

print("\n🔹 Shape before removing duplicates:", df.shape)
df.drop_duplicates(inplace=True)
print("🔹 Shape after removing duplicates:", df.shape)

df['sex'] = df['sex'].str.lower().str.strip()
df['embarked'] = df['embarked'].replace({'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'})

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])          
df['embarked'] = le.fit_transform(df['embarked'])

df = pd.get_dummies(df, columns=['class'], drop_first=True)

scaler = StandardScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['age'] >= lower) & (df['age'] <= upper)]

z_scores = stats.zscore(df['fare'])
df = df[(abs(z_scores) < 3)]

df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)
df['age_group'] = pd.cut(df['age'],
                         bins=[-1, -0.7, 0, 0.5, 1.0, 2.0],
                         labels=['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior'])

print("\n🔹 Cleaned Dataset Info:")
print(df.info())

print("\n🔹 Final 5 Rows:")
print(df.head())

print("\n✅ Data Cleaning and Transformation Completed Successfully!")
