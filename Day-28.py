import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats


df = sns.load_dataset('titanic')
print("✅ Titanic dataset loaded successfully")
print(df.head())

print("\n--- Basic Information ---")
print(df.info())
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

df.drop(['deck', 'embark_town', 'alive', 'adult_male', 'who', 'class'], axis=1, inplace=True)

df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

print("\n✅ Missing values handled successfully")
print(df.isnull().sum())

sns.set_theme(style="whitegrid", palette="Set2")

plt.figure(figsize=(6,4))
sns.countplot(x='sex', data=df)
plt.title("Gender Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['fare'], bins=30, kde=True, color='orange')
plt.title("Fare Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='pclass', data=df)
plt.title("Passenger Class Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='sex', y='survived', data=df, palette='coolwarm')
plt.title("Survival Rate by Gender")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='pclass', y='survived', data=df, palette='muted')
plt.title("Survival Rate by Passenger Class")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='survived', y='age', data=df, palette='pastel')
plt.title("Age vs Survival")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='survived', y='fare', data=df, palette='husl')
plt.title("Fare vs Survival")
plt.show()

plt.figure(figsize=(7,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[['age', 'fare', 'pclass', 'survived']], hue='survived', diag_kind='kde')
plt.suptitle("Pairplot - Multivariate Analysis", y=1.02)
plt.show()


df['age_group'] = pd.cut(df['age'], bins=[0,12,18,35,60,80],
                         labels=['Child','Teen','Young Adult','Adult','Senior'])

df['family_size'] = df['sibsp'] + df['parch'] + 1


encoder = LabelEncoder()
df['sex_encoded'] = encoder.fit_transform(df['sex'])
df['embarked_encoded'] = encoder.fit_transform(df['embarked'])

plt.figure(figsize=(7,5))
sns.barplot(x='age_group', y='survived', data=df, palette='viridis')
plt.title("Survival Rate by Age Group")
plt.show()

plt.figure(figsize=(7,5))
sns.barplot(x='family_size', y='survived', data=df, palette='coolwarm')
plt.title("Survival Rate by Family Size")
plt.show()

plt.figure(figsize=(7,5))
sns.barplot(x='embarked', y='survived', hue='sex', data=df, palette='muted')
plt.title("Survival Rate by Embarkation & Gender")
plt.show()

z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
outliers = (z_scores > 3).sum().sum()
print(f"\n🔍 Outlier count (Z-score > 3): {outliers}")


df.to_csv("titanic_eda_final.csv", index=False)
print("\n✅ Cleaned Titanic dataset saved as 'titanic_eda_final.csv'")
