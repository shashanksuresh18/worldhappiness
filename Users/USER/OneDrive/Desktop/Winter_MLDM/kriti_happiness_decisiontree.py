# Decision tree: GridSearch

# Data Preparation and splitting into x and y train



import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

bins = [happiness_2016_cleaned['Happiness Score'].min(), 4.5, 6.0, happiness_2016_cleaned['Happiness Score'].max()]
labels = ['Low', 'Medium', 'High']
happiness_2016_cleaned['Happiness Category'] = pd.cut(happiness_2016_cleaned['Happiness Score'], bins=bins, labels=labels)

happiness_2016_cleaned = happiness_2016_cleaned.dropna(subset=['Happiness Category'])

X = happiness_2016_cleaned[['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 
                            'Freedom', 'Trust (Government Corruption)', 'Generosity', 
                            'Dystopia Residual', 'Economy_Health_Index']]
y = happiness_2016_cleaned['Happiness Category']

print(X.isnull().sum())
print(y.isnull().sum())

y_binarized = label_binarize(y, classes=labels)

X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.2, random_state=42)


# Model Training and Evaluation Before Hyperparameter Tuning



from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dtc = DecisionTreeClassifier(random_state=42)

dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

accuracy_before_tuning = accuracy_score(y_test, y_pred)

print("Accuracy before Hyperparameter Tuning:", accuracy_before_tuning)


# Model Training and accuracy after Hyperparameter Tuning  



from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}

dtc = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_dtc = grid_search.best_estimator_

y_pred = best_dtc.predict(X_test)

accuracy_after_tuning = accuracy_score(y_test, y_pred)

print("Accuracy after Hyperparameter Tuning:", accuracy_after_tuning)


# Evaluation and Visualization, including accuracy 



from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
class_report = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=labels)

print('Best Parameters:', grid_search.best_params_)
print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_binarized.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = roc_auc_score(y_test[:, i], y_pred[:, i])

plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {labels[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

