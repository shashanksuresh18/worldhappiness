import gym
from gym import spaces
import numpy as np

class HealthEnv(gym.Env):
    def __init__(self, X, y):
        super(HealthEnv, self).__init__()
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.current_sample = 0
        self.observation_space = spaces.Discrete(self.n_samples)
        self.action_space = spaces.Discrete(2)  # Binary classification: 0 (no stroke), 1 (stroke)

    def reset(self):
        self.current_sample = 0
        return self.current_sample

    def step(self, action):
        true_label = self.y.iloc[self.current_sample]
        reward = 1 if action == true_label else -1
        self.current_sample += 1
        done = self.current_sample >= self.n_samples
        next_state = self.current_sample if not done else 0  # Ensure next state is valid
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

def Q_learning(env, step_size, episodes, gamma, epsilon, min_epsilon, decay_rate):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    success_episodes = 0

    for episode in range(episodes):
        state = env.reset()
        total_R = 0  # Reward count for each episode
        while True:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _ = env.step(action)
            total_R += reward

            if reward == 1 and done:
                success_episodes += 1

            if done:
                break

            old_value = Q[state, action]
            next_max = np.max(Q[next_state]) if next_state is not None else 0
            new_value = old_value + step_size * (reward + gamma * next_max - old_value)
            Q[state, action] = new_value

            state = next_state

        epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-decay_rate * episode)

        # Print results for each episode
        print(f"Episode {episode}: ||  Total Reward: {total_R} || e-greedy: {epsilon} || Success Rate: {success_episodes}/{episode+1}")

    return Q, success_episodes

def evaluate_policy(env, policy):
    y_pred = []
    y_pred_prob = []

    state = env.reset()
    for _ in range(env.n_samples):
        action = policy[state]
        y_pred.append(action)
        next_state, reward, done, _ = env.step(action)
        y_pred_prob.append(np.max(Q[state]))  # Append the probability score
        if done:
            break
        state = next_state

    return np.array(y_pred), np.array(y_pred_prob)

# Hyperparameters grid
from itertools import product

step_sizes = [0.1, 0.5]
episodes_list = [100, 200]
gammas = [0.8, 0.95]
min_epsilons = [0.01, 0.1]
decay_rates = [0.001, 0.01]

best_roc_auc = 0
best_params = {}

# Grid search for hyperparameter tuning
for step_size, episodes, gamma, min_epsilon, decay_rate in product(step_sizes, episodes_list, gammas, min_epsilons, decay_rates):
    env = HealthEnv(X_train, y_train)
    Q, success_episodes = Q_learning(env, step_size, episodes, gamma, 1.0, min_epsilon, decay_rate)
    policy = np.argmax(Q, axis=1)

    env_test = HealthEnv(X_test, y_test)
    y_pred, y_pred_prob = evaluate_policy(env_test, policy)

    roc_auc = roc_auc_score(y_test, y_pred_prob)

    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_params = {
            'step_size': step_size,
            'episodes': episodes,
            'gamma': gamma,
            'min_epsilon': min_epsilon,
            'decay_rate': decay_rate
        }

# Output the best hyperparameters
print(f"Best ROC AUC: {best_roc_auc}")
print(f"Best Parameters: {best_params}")

# Train the final model with the best hyperparameters
env = HealthEnv(X_train, y_train)
Q, success_episodes = Q_learning(env, best_params['step_size'], best_params['episodes'], best_params['gamma'], 1.0, best_params['min_epsilon'], best_params['decay_rate'])
policy = np.argmax(Q, axis=1)

env_test = HealthEnv(X_test, y_test)
y_pred, y_pred_prob = evaluate_policy(env_test, policy)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Q-learning-like Model Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n{cm}")
print(f"Classification Report:\n{report}")

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Q-learning-like Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
