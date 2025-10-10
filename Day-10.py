import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Month": ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"],
    "Sales": [12000, 15000, 18000, 22000, 26000, 30000, 28000, 31000, 34000, 37000, 39000, 42000]
}

df = pd.DataFrame(data)

df.to_csv("monthly_sales.csv", index=False)

df = pd.read_csv("monthly_sales.csv")

plt.figure(figsize=(10, 6))
plt.plot(df["Month"], df["Sales"], marker='o', linestyle='-', linewidth=2)

plt.title("Monthly Sales Growth in 2025", fontsize=16, fontweight='bold', color='darkblue')
plt.xlabel("Month", fontsize=12)
plt.ylabel("Sales (in USD)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
