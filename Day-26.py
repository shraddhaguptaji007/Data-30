import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

df = sns.load_dataset('titanic')

print(df.head())
print(df.info())

df = df.drop(['deck', 'embark_town', 'alive', 'adult_male', 'class', 'who'], axis=1)

df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

print("\n✅ Cleaned Dataset:")
print(df.head())

sns.pairplot(df[['age', 'fare', 'survived', 'pclass']], hue='survived', diag_kind='kde')
plt.suptitle("Pairplot - Multivariate Relationships", y=1.02)
plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

pivot = pd.pivot_table(df, values='survived', index='sex', columns='pclass', aggfunc='mean')
print("\n📊 Survival Rate Pivot Table:")
print(pivot)
sns.heatmap(pivot, annot=True, cmap='YlGnBu')
plt.title("Survival Rate by Gender and Class")
plt.show()


df['log_fare'] = np.log1p(df['fare'])

plt.figure(figsize=(6,4))
sns.histplot(df['log_fare'], kde=True, color='purple')
plt.title("Log-Transformed Fare Distribution")
plt.show()

df['age_group'] = pd.cut(df['age'], bins=[0,12,18,35,60,80],
                         labels=['Child','Teen','Young Adult','Adult','Senior'])
print("\n🧩 Age Group Distribution:")
print(df['age_group'].value_counts())

sns.countplot(x='age_group', hue='survived', data=df)
plt.title("Survival by Age Group")
plt.show()

encoder = LabelEncoder()
df['sex_encoded'] = encoder.fit_transform(df['sex'])
df['embarked_encoded'] = encoder.fit_transform(df['embarked'])

numeric_cols = ['age', 'fare', 'pclass', 'sex_encoded', 'embarked_encoded']
X = df[numeric_cols].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=df['survived'])
plt.title("PCA Visualization - Titanic Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

print("\nPCA Explained Variance Ratio:", pca.explained_variance_ratio_)
