# Data Loading:



import pandas as pd
import numpy as np

# Load the Happiness datasets
happiness_2016 = pd.read_csv('happiness_2016.csv')

# Display the first few rows to understand the structure of the data
print(happiness_2016.head())


# Data Cleaning:



happiness_2016 = pd.read_csv('happiness_2016.csv')

# Check for missing values
missing_values = happiness_2016.isnull().sum()
print(missing_values)

# but in this dataset there are no missing values
happiness_2016_cleaned = happiness_2016.copy()


# Data Integration:



happiness_2017 = pd.read_csv('happiness_2017.csv')

# Merge the datasets on a common column (e.g., Country)
happiness_combined = pd.merge(happiness_2016_cleaned, happiness_2017, on='Country', suffixes=('_2016', '_2017'))

print(happiness_combined.head())


# Variable Transformation:



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

happiness_2016_cleaned['Happiness Score'] = scaler.fit_transform(happiness_2016_cleaned[['Happiness Score']])

print(happiness_2016_cleaned.head())


# Dimensionality Reduction:




from sklearn.decomposition import PCA

numeric_columns = happiness_2016_cleaned.select_dtypes(include=[np.number])

pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
principal_components = pca.fit_transform(numeric_columns)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

happiness_2016_with_pca = pd.concat([happiness_2016_cleaned.reset_index(drop=True), pca_df], axis=1)

print(happiness_2016_with_pca.head())
