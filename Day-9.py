import pandas as pd
import numpy as np

data = {
    "Name": ["Amit", "Priya", "Rohan", "Neha", "Vikram", "Anita", "Raj", "Sonia", "Karan", "Meena"],
    "Class": ["A", "A", "B", "B", np.nan, "A", "B", "C", "C", "C"],
    "Marks": [88, np.nan, 79, 85, 95, 67, np.nan, 89, 91, 84]
}

df = pd.DataFrame(data)

df.to_csv("student_marks_with_missing.csv", index=False)

df = pd.read_csv("student_marks_with_missing.csv")
print("Original Dataset:\n", df, "\n")

clean_df = df.dropna()
print("Cleaned Dataset:\n", clean_df, "\n")

grouped_df = clean_df.groupby("Class")["Marks"].mean().reset_index()
print("Average Marks by Class:\n", grouped_df)
