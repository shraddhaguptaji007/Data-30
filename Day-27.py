import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset('titanic')

df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(16, 10))
plt.suptitle("🚢 Titanic Data Visualization Dashboard", fontsize=18, fontweight='bold', y=1.02)

plt.subplot(2, 3, 1)
sns.countplot(x='sex', hue='survived', data=df, palette='Set2')
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", loc='upper right')

plt.subplot(2, 3, 2)
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.subplot(2, 3, 3)
sns.barplot(x='pclass', y='survived', data=df, palette='coolwarm')
plt.title("Average Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.subplot(2, 3, 4)
sns.boxplot(x='pclass', y='fare', data=df, palette='husl')
plt.title("Fare Distribution by Class")
plt.xlabel("Class")
plt.ylabel("Fare")

plt.subplot(2, 3, 5)
sns.barplot(x='embarked', y='survived', hue='sex', data=df, palette='muted')
plt.title("Survival Rate by Embarkation Point")
plt.xlabel("Embarkation")
plt.ylabel("Survival Rate")

plt.subplot(2, 3, 6)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n📊 Key Insights from Visualization:")
print("1️⃣ Females had significantly higher survival rates than males.")
print("2️⃣ 1st-class passengers had the highest survival rate; 3rd class had the lowest.")
print("3️⃣ Most passengers were between 20–40 years old.")
print("4️⃣ Higher fares generally correlate with better survival chances.")
print("5️⃣ Embarkation point 'C' had the highest survival rate overall.")
