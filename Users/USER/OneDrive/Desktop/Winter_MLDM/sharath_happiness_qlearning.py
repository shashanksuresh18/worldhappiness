
# Q-learning in a Grid World

# Data Preparation
# 



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

happiness_2016 = pd.read_csv('happiness_2016.csv')

happiness_2016_cleaned = happiness_2016.copy()

happiness_2016_cleaned['Economy_Health_Index'] = happiness_2016_cleaned['Economy (GDP per Capita)'] + happiness_2016_cleaned['Health (Life Expectancy)']

happiness_2016_cleaned['GDP_Family_Index'] = happiness_2016_cleaned['Economy (GDP per Capita)'] + happiness_2016_cleaned['Family']
happiness_2016_cleaned['Freedom_Trust_Index'] = happiness_2016_cleaned['Freedom'] + happiness_2016_cleaned['Trust (Government Corruption)']

bins = [happiness_2016_cleaned['Happiness Score'].min(), 4.5, 6.0, happiness_2016_cleaned['Happiness Score'].max()]
labels = ['Low', 'Medium', 'High']
happiness_2016_cleaned['Happiness Category'] = pd.cut(happiness_2016_cleaned['Happiness Score'], bins=bins, labels=labels)

happiness_2016_cleaned = happiness_2016_cleaned.dropna(subset=['Happiness Category'])

X = happiness_2016_cleaned[['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 
                            'Freedom', 'Trust (Government Corruption)', 'Generosity', 
                            'Dystopia Residual', 'Economy_Health_Index', 'GDP_Family_Index', 'Freedom_Trust_Index']]
y = happiness_2016_cleaned['Happiness Category']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

n_clusters = 30  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
state_labels_train = kmeans.fit_predict(X_train)
state_labels_test = kmeans.predict(X_test)


# Simple Environment Definition
# 



from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class SimpleEnv:
    def __init__(self, data, state_labels, true_labels):
        self.data = data
        self.state_labels = state_labels
        self.true_labels = pd.Series(true_labels).reset_index(drop=True)  
        self.n_states = np.unique(state_labels).size
        self.n_actions = data.shape[1]
        self.current_state_idx = 0
        self.current_state = state_labels[self.current_state_idx]
        self.observation_space = np.arange(self.n_states)
        self.action_space = np.arange(self.n_actions)
    
    def reset(self):
        self.current_state_idx = np.random.randint(0, len(self.state_labels))
        self.current_state = self.state_labels[self.current_state_idx]
        return self.current_state, {}
    
    def step(self, action):
        state_value = self.data[self.current_state_idx, int(action)] + np.random.normal(0, 0.5)
        reward = 0
        actual_class = self.true_labels.iloc[self.current_state_idx]

        if state_value > 0.5:
            if actual_class == 'High':
                reward = 3.0
            elif actual_class == 'Medium':
                reward = 2.0
            else:
                reward = 1.0
        else:
            reward = -0.5  

        self.current_state_idx = (self.current_state_idx + 1) % len(self.state_labels)
        self.current_state = self.state_labels[self.current_state_idx]
        done = self.current_state_idx == 0
        return self.current_state, reward, done, False, {}

def decay_epsilon(episode: int, max_epsilon: float = 1.0, min_epsilon: float = 0.1, decay_rate: float = 0.00005) -> float:
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    return epsilon


# Q-learning Algorithm




def Q_learning(env: SimpleEnv, step_size: float = 0.8, episodes: int = 20000, gamma: float = 0.9, success_threshold: float = 2.0) -> np.ndarray:
    Q = np.zeros([env.n_states, env.n_actions])
    epsilon = 1.0  
    success_episodes = 0

    for i in range(episodes):
        state, _ = env.reset()  
        total_R = 0 

        while True: 
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, env.n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, truncated, _ = env.step(action)
            total_R += reward

            Q[state][action] += step_size * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

            state = next_state

            if done or truncated:
                break

        if total_R >= success_threshold:
            success_episodes += 1

        epsilon = decay_epsilon(i)

        if (i + 1) % 1000 == 0 or i == episodes - 1:
            print(f"Episode {i + 1}: || Total Reward: {total_R:.2f} || e-greedy: {epsilon:.2f} || Success Rate: {success_episodes}/{i + 1}")

    return Q


# Policy Determination
# 



def determine_policy_from_Q(Q: np.ndarray) -> np.ndarray:
    policy = np.zeros(Q.shape[0], dtype=int)
    for s in range(Q.shape[0]):
        policy[s] = np.argmax(Q[s])
    return policy

def test_policy(env, policy, success_threshold: float = 2.0, runs=100):
    total_reward = 0
    success_count = 0
    y_true = []
    y_scores = []
    all_rewards = []  
    for _ in range(runs):
        state, _ = env.reset()
        run_reward = 0
        while True:
            action = policy[state]
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            run_reward += reward
            y_true.append(env.true_labels.iloc[env.current_state_idx])
            y_scores.append(reward) 
            state = next_state
            if done:
                break
        all_rewards.append(run_reward)
        if run_reward >= success_threshold:
            success_count += 1
    return total_reward / runs, success_count / runs, y_true, y_scores, all_rewards


# Main Execution and Evaluation and displaying accuracy before and after hyperparameter tuning
# 



from sklearn.metrics import accuracy_score, confusion_matrix


if __name__ == "__main__":
    SUCCESS_REWARD = 2.0  
    env_train = SimpleEnv(X_train, state_labels_train, y_train)
    env_test = SimpleEnv(X_test, state_labels_test, y_test)

    Q_before = Q_learning(env_train, step_size=0.8, episodes=10000, gamma=0.9, success_threshold=SUCCESS_REWARD)
    opt_p_before = determine_policy_from_Q(Q_before)
    avg_reward_before, success_rate_before, y_true_before, y_scores_before, all_rewards_before = test_policy(env_test, opt_p_before, success_threshold=SUCCESS_REWARD)
    accuracy_before = accuracy_score([1 if label == 'High' else 0 for label in y_true_before], [1 if score > 1.0 else 0 for score in y_scores_before])
    print(f"Before Hyperparameter Tuning - Average Reward: {avg_reward_before:.2f}, Success Rate: {success_rate_before * 100:.2f}%, Accuracy: {accuracy_before * 100:.2f}%")

    Q_after = Q_learning(env_train, step_size=2.8, episodes=20000, gamma=0.99, success_threshold=SUCCESS_REWARD)
    opt_p_after = determine_policy_from_Q(Q_after)
    avg_reward_after, success_rate_after, y_true_after, y_scores_after, all_rewards_after = test_policy(env_test, opt_p_after, success_threshold=SUCCESS_REWARD)
    accuracy_after = accuracy_score([1 if label == 'High' else 0 for label in y_true_after], [1 if score > 1.0 else 0 for score in y_scores_after])
    print(f"After Hyperparameter Tuning - Average Reward: {avg_reward_after:.2f}, Success Rate: {success_rate_after * 100:.2f}%, Accuracy: {accuracy_after * 100:.2f}%")



# Confusion matrix, ROC Curve and Learning Curve



from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_true_labels = pd.cut([score for score in y_scores_after], bins=[-np.inf, 0.5, 1.5, np.inf], labels=labels)
y_pred_labels = pd.cut([score for score in y_scores_after], bins=[-np.inf, 0.5, 1.5, np.inf], labels=labels)

y_true_labels = y_true_labels[:len(y_pred_labels)]

y_binarized_test = label_binarize(y_true_labels, classes=labels)
y_binarized_scores = label_binarize(y_pred_labels, classes=labels)

conf_matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

n_classes = y_binarized_test.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_binarized_test[:, i], y_binarized_scores[:, i])
    roc_auc[i] = roc_auc_score(y_binarized_test[:, i], y_binarized_scores[:, i])

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

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(range(len(all_rewards_after)), all_rewards_after, color='blue', lw=2)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Learning Curve')
plt.show()