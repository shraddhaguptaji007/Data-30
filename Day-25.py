import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')

print(df.head())

num_cols = ['age', 'fare']

plt.figure(figsize=(7, 5))
sns.scatterplot(x='age', y='fare', data=df, hue='class', palette='viridis')
plt.title('Age vs Fare by Class', fontsize=13)
plt.show()

sns.pairplot(df[['age', 'fare', 'survived']], hue='survived', palette='coolwarm', diag_kind='kde')
plt.suptitle("Pairplot between Age, Fare, and Survival", y=1.02)
plt.show()

corr_matrix = df[['age', 'fare', 'survived', 'pclass']].corr()

print("\n📊 Correlation Matrix:")
print(corr_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=13)
plt.show()

high_corr = corr_matrix[(corr_matrix > 0.8) & (corr_matrix < 1.0)]
print("\n🔍 Highly Correlated Features:")
print(high_corr.dropna(how='all').dropna(axis=1, how='all'))

plt.figure(figsize=(7, 5))
sns.boxplot(x='class', y='fare', data=df, palette='pastel')
plt.title('Fare Distribution Across Passenger Classes', fontsize=13)
plt.show()

cross_tab = pd.crosstab(df['sex'], df['survived'], normalize='index') * 100
print("\n📊 Survival Rate by Gender (%):")
print(cross_tab)

cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title("Survival Percentage by Gender")
plt.xlabel("Sex")
plt.ylabel("Percentage")
plt.show()
