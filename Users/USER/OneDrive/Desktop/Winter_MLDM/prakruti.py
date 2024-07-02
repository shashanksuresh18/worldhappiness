# KNN

# Model Training and Evaluation Before Hyperparameter Tuning



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

accuracy_before_tuning = accuracy_score(y_test, y_pred)

print("Accuracy before Hyperparameter Tuning:", accuracy_before_tuning)


# Hyperparameter Tuning with GridSearchCV



from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_knn = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)


# Training KNN Model After Hyperparameter Tuning



from sklearn.metrics import accuracy_score

y_pred_tuned = best_knn.predict(X_test)

accuracy_after_tuning = accuracy_score(y_test, y_pred_tuned)

print("Accuracy after Hyperparameter Tuning:", accuracy_after_tuning)


# Evaluation and Visualization



conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_tuned.argmax(axis=1))
class_report = classification_report(y_test.argmax(axis=1), y_pred_tuned.argmax(axis=1), target_names=labels)

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
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_tuned[:, i])
    roc_auc[i] = roc_auc_score(y_test[:, i], y_pred_tuned[:, i])

plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {labels[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()