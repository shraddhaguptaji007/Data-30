import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

tips = sns.load_dataset("tips")

print("Dataset Info:")
print(tips.info())
print("\nFirst 5 Rows:")
print(tips.head())

print("\nAvailable Seaborn Datasets:")
print(sns.get_dataset_names())

sns.set_style("whitegrid")      
sns.set_palette("coolwarm")        

plt.figure(figsize=(8, 6))
sns.boxplot(x="day", y="tip", data=tips)

plt.title("Distribution of Tips by Day", fontsize=16, fontweight='bold', color='darkblue')
plt.xlabel("Day of the Week", fontsize=12)
plt.ylabel("Tip Amount ($)", fontsize=12)
plt.tight_layout()

plt.show()
