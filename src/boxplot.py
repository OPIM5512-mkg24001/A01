from sklearn.datasets import fetch_california_housing
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Create the figs directory if it doesn't exist
output_dir = 'figs'
os.makedirs(output_dir, exist_ok=True)

# Make a boxplot of 'MedInc' (Median Income)
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['MedInc'])
plt.title('Boxplot of Median Income (MedInc)')
plt.ylabel('Median Income (tens of thousands of dollars)')

# Save the figure
output_path = os.path.join(output_dir, 'boxplot.png')
plt.savefig(output_path)
plt.show()

print(f"Boxplot saved to {output_path}")