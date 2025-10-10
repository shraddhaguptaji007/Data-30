import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

print("Dataset Info:")
print(iris.info())
print("\nFirst 5 Rows:\n", iris.head())
print("\nSummary Statistics:\n", iris.describe())

print("\nMissing Values in Each Column:\n", iris.isnull().sum())

iris.rename(columns={
    "sepal_length": "Sepal_Length",
    "sepal_width": "Sepal_Width",
    "petal_length": "Petal_Length",
    "petal_width": "Petal_Width"
}, inplace=True)

sns.pairplot(iris, hue="species")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x="species", y="Petal_Length", data=iris, palette="Set2")
plt.title("Distribution of Petal Length by Species", fontsize=14)
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(iris.drop(columns="species").corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Iris Features")
plt.show()

