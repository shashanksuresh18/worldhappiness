# Derivation of New Variables:



happiness_2016_cleaned['Economy_Health_Index'] = happiness_2016_cleaned['Economy (GDP per Capita)'] + happiness_2016_cleaned['Health (Life Expectancy)']

print(happiness_2016_cleaned.head())


# Data Visualisation: Pair plot



import seaborn as sns
import matplotlib.pyplot as plt


happiness_2016 = pd.read_csv('happiness_2016.csv')

happiness_2016_cleaned = happiness_2016.copy()

happiness_2016_cleaned['Economy_Health_Index'] = happiness_2016_cleaned['Economy (GDP per Capita)'] + happiness_2016_cleaned['Health (Life Expectancy)']

sns.pairplot(happiness_2016_cleaned, diag_kind='hist', plot_kws={'alpha':0.5})

plt.show()


# Data Visualisation: Box plot



numeric_columns = happiness_2016_cleaned.select_dtypes(include=[np.number])

plt.figure(figsize=(15, 10))

for i, column in enumerate(numeric_columns.columns, 1):
    plt.subplot(3, 4, i)  # Adjust the subplot grid if necessary
    sns.boxplot(data=happiness_2016_cleaned, y=column)
    plt.title(f'Box Plot of {column}')

plt.tight_layout()
plt.show()





correlation_matrix = happiness_2016_cleaned.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
