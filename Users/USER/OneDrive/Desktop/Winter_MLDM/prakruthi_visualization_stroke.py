import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='hypertension', y='age', data=df)
plt.title('Boxplot of Age by Hypertension')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='stroke', y='avg_glucose_level', data=df)
plt.title('Boxplot of Average Glucose Level by Stroke')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='stroke', y='bmi', data=df)
plt.title('Boxplot of BMI by Stroke')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='hypertension', y='age', data=df)
plt.title('Violin Plot of Age by Hypertension')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='stroke', y='avg_glucose_level', data=df)
plt.title('Violin Plot of Average Glucose Level by Stroke')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='stroke', y='bmi', data=df)
plt.title('Violin Plot of BMI by Stroke')
plt.show()

numeric_features = ['age', 'avg_glucose_level', 'bmi']

sns.pairplot(df[numeric_features + ['stroke']], hue='stroke', palette='Set1')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['avg_glucose_level'], bins=20, kde=True)
plt.title('Average Glucose Level Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['bmi'], bins=20, kde=True)
plt.title('BMI Distribution')
plt.show()

plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print("Data Exploration and Visualization completed.")
