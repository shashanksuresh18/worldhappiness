
# Naive Bayes with One-vs-Rest

# Training Naive Bayes Model Before Hyperparameter Tuning



from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

nb_model = OneVsRestClassifier(GaussianNB())
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

accuracy_before_tuning = accuracy_score(y_test, y_pred)

print("Accuracy before Hyperparameter Tuning:", accuracy_before_tuning)


# Training Naive Bayes Model After Hyperparameter Tuning



from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

param_grid = {
    'estimator__var_smoothing': np.logspace(0, -9, num=100)  # GaussianNB specific parameter
}

grid_search = GridSearchCV(estimator=OneVsRestClassifier(GaussianNB()), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_nb = grid_search.best_estimator_


y_pred_tuned = best_nb.predict(X_test)

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