# Data Exploration

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('data/adult.csv')

# 1. List the names of all numerical attributes in the dataset
numerical_attributes = df.select_dtypes(include=['number']).columns
print("Numerical attributes:", numerical_attributes)

# 2. Count the unique values present in the dataset for each attribute
unique_values_counts = df.nunique()
print("\nNumber of unique values for each attribute:\n", unique_values_counts)

# 3. List the unique values of the 'workclass' attribute
workclass_unique_values = df['workclass'].unique()
print("\nUnique values of 'workclass' attribute:\n", workclass_unique_values)

# 4. List all attributes with missing values and the number of missing values
missing_values = df.isnull().sum()
missing_values_2 = (df == "?").sum()
missing_values = missing_values[missing_values > 0]
missing_values_2 = missing_values_2[missing_values_2 > 0]
print("\nAttributes with missing values:\n", missing_values)
print("\nAttributes with missing values (?):\n", missing_values_2)

# 5. Calculate the percentage of individuals who are natively from the United States
total_individuals = len(df)
us_natives = len(df[df['native-country'] == 'United-States'])
percentage_us_natives = (us_natives / total_individuals) * 100
print("\nPercentage of individuals who are natively from the United States: {:.2f}%".format(percentage_us_natives))

# 6. Create a bar plot for the 'native-country' attribute
country_counts = df['native-country'].value_counts()
country_counts = country_counts[country_counts.index != 'United-States']  # Exclude US for clarity
plt.figure(figsize=(10, 6))
plt.bar(country_counts.index, country_counts.values)
plt.xlabel("Native Country")
plt.ylabel("Number of Individuals")
plt.title("Bar Plot of Native Country (excluding US)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()