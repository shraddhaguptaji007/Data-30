import numpy as np
from scipy import stats


group1 = np.random.normal(50, 5, 100)   
group2 = np.random.normal(52, 5, 100)   

difference = group2 - group1  

matrix_A = np.random.randint(1, 10, (3, 3))
matrix_B = np.random.randint(1, 10, (3, 3))
matrix_product = np.dot(matrix_A, matrix_B)

print("Matrix A:\n", matrix_A)
print("\nMatrix B:\n", matrix_B)
print("\nMatrix Product (A · B):\n", matrix_product)

t_stat, p_value = stats.ttest_ind(group1, group2)

print("\nT-Test for Mean Comparison between Group1 and Group2:")
print("T-Statistic =", round(t_stat, 3))
print("P-Value =", round(p_value, 4))

if p_value < 0.05:
    print("✅ Statistically Significant Difference (reject H0)")
else:
    print("❌ No Significant Difference (fail to reject H0)")

corr_coeff, corr_p = stats.pearsonr(group1, group2)
print("\nPearson Correlation Test:")
print("Correlation Coefficient =", round(corr_coeff, 3))
print("P-Value =", round(corr_p, 4))

