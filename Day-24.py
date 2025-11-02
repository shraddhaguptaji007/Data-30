import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')

print(df.info())
print(df.describe())


numerical_cols = ['age', 'fare']

for col in numerical_cols:
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}', fontsize=13)

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}', fontsize=13)

    plt.show()

    print(f"\n📊 Summary for '{col}':")
    print(df[col].describe())
    print(f"Skewness: {df[col].skew():.2f}")
    print(f"Kurtosis: {df[col].kurt():.2f}")
    print("-" * 50)


categorical_cols = ['sex', 'class', 'embarked', 'alive']

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[col], palette='pastel')
    plt.title(f'Count of {col}', fontsize=13)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

    print(f"\n📊 Value Counts for '{col}':")
    print(df[col].value_counts(normalize=True) * 100)
    print("-" * 50)
