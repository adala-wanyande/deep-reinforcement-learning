# Experiments

## Overview

This project consists of two primary ablation studies:

1. **Hyperparameter Investigation (Naive DQN)**
2. **Improvements for Stability (Stable DQN)**

Each experiment was run over multiple independent trials to ensure statistical reliability and to better understand the impact of different factors on performance.

---

## **Experiment 1 - Naive Deep Q-Network (DQN)**

### **Objective:**
The goal was to implement a basic DQN and study the impact of different hyperparameters on its performance.

### **Implementation Details:**
- **Runs per setting:** 5 (to reduce variance and improve result consistency)
- **Environment:** `CartPole-v1`
- **Total steps per run:** 1,000,000
- **Initial hyperparameters:**
  
```python
NUM_RUNS = 5
MAX_ENV_STEPS = int(1e6)
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
```

- **Network Architecture:**
  
```python
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
```

### **Findings:**
- **Performance plateaued at 200 reward** (indicating limited learning capability).
- **Agent failed to generalize beyond simple policies.**
- **Possible causes of poor performance:**
  - Insufficient network complexity.
  - Poor exploration due to fast epsilon decay.
  - Learning rate instability.

### **Hyperparameter Tuning & Analysis:**

#### **1. Learning Rate Variations**
- **Small (`0.0001`)**: Slow learning, agent fails early (10-50 steps per episode).
- **Medium (`0.0005`)**: Balanced learning but still limited performance.
- **High (`0.001`)**: Fastest learning, highest overall reward achieved.

#### **2. Discount Factor (Gamma) Variations**
- **Low (`0.85`)**: Overemphasizes immediate rewards, fails long-term strategies.
- **Medium (`0.95`)**: Balanced short-term and long-term planning.
- **High (`0.99`)**: Encourages long-term rewards, performed best.

#### **3. Exploration Rate (Epsilon) Variations**
- **High (`0.9999`)**: Agent explores too long, learns suboptimal strategies.
- **Medium (`0.999`)**: Balanced exploration-exploitation tradeoff, best performance.
- **Low (`0.99`)**: Overfits too quickly, fails to learn optimal policies.

#### **4. Neural Network Size Variations**
- **Small (`[24-1]`)**: Lacked complexity, reached 200 reward but no improvements.
- **Medium (`[64-64-2]`)**: Best performance, stable training.
- **Large (`[128-128-128-3]`)**: More stable, but diminishing returns in performance.

### **Conclusion:**
- Despite hyperparameter tuning, the **Naive DQN struggled with stability**.
- Instability arose from **bootstrapping, off-policy learning, and function approximation issues**.
- **Next step:** Explore stabilization techniques like **experience replay** and **target networks**.

---

## **Experiment 2 - Ablation Study on Stable DQN**

### **Objective:**
To test key stabilization techniques individually and in combination:

1. **Experience Replay Only**
2. **Target Networks Only**
3. **Both Experience Replay & Target Networks**

### **Key Modifications:**
- **Experience Replay:** Stores past experiences in a buffer and samples minibatches to break correlation in updates.
- **Target Network:** Uses a delayed update network to stabilize learning.
- **Final Settings:**

```python
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 0.99
EPSILON_DECAY = 0.999
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 1000
```

### **Findings:**

#### **1. Experience Replay Only**
- More stable training than Naive DQN.
- Helped break correlation in training updates.
- **Still showed divergence issues over long runs.**

#### **2. Target Network Only**
- Reduced **training variance** significantly.
- **Training was still unstable**, but better than Naive DQN.
- Did not fully prevent overfitting.

#### **3. Both Experience Replay & Target Network**
- **Best overall performance.**
- **Converged faster, with reduced variance.**
- **Highest reward achieved**, sustaining beyond 200 steps consistently.

### **Final Conclusion:**
- **Stable DQN significantly outperformed Naive DQN.**
- **Target Networks helped reduce variance.**
- **Experience Replay improved stability and training efficiency.**
- **Combining both yielded the most stable and high-performing agent.**

**(Plots & additional findings will be included as supporting evidence.)**

