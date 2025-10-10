import pandas as pd

df = pd.read_csv("student_marks.csv")

print("Basic Info:")
print(df.info())

print("\nTop 5 Rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())
