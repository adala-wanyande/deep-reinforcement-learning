# Deep Reinforcement Learning – MSc Course (LIACS)

This repository contains all code, experiments, reports, and analysis from my work on the Deep Reinforcement Learning (DRL) course at Leiden University.

## Structure

```bash
DEEP-RL/
│
├── assignment-1/               # Assignment 1: Q-Learning & Deep Q-Networks
│   ├── naive-dqn/              # Naive DQN implementation
│   ├── stable-dqn/             # Target networks, experience replay
│   ├── bonus/                  # Optional enhancements
│   ├── visuals/                # Plots and visualizations
│   ├── final_report.pdf        # PDF report for A1
│   ├── plot_all.py             # Plotting script
│   └── run_ablation_studies.sh
│
├── assignment-2/               # Assignment 2: Policy Gradient Algorithms
│   ├── data/                   # Experiment logs (CSV)
│   ├── plots/                  # Generated graphs
│   ├── report/                 # LaTeX source for ICML-style report
│   ├── hyperparameter-tuning/ # Ablation scripts
│   ├── submission/             # Clean submission copy
│   ├── reinforce.py            # REINFORCE implementation
│   ├── ac.py                   # Actor-Critic (vanilla)
│   ├── a2c.py                  # Advantage Actor-Critic (A2C)
│   ├── models.py               # Shared model architectures
│   ├── plot_all.py             # Plotting script (CartPole)
│   ├── plot_all_bonus.py       # Plotting script (Acrobot)
│   └── run_all.sh              # Run all experiments
│
└── README.md                   # You are here
```

---

## Assignment Overview

### **Assignment 1 – Value-Based Methods**
- Implemented tabular Q-learning and Deep Q-Networks (DQN).
- Added experience replay and target networks.
- Compared naive vs stabilized DQN setups.

### **Assignment 2 – Policy-Based Methods**
- Implemented REINFORCE, Actor-Critic, and Advantage Actor-Critic (A2C).
- Trained and evaluated on `CartPole-v1` and `Acrobot-v1`.
- Performed ablation studies on learning rate and critic capacity.
- Included comparison with DQN for reference.

---

## Running Experiments

### Setup Environment

```bash
python -m venv drl-env
source drl-env/bin/activate  # Windows: drl-env\Scripts\activate
```

### Install Requirements

Each assignment folder has its own `requirements.txt`:

```bash
cd assignment-2/
pip install -r requirements.txt
```

### Run Experiments

Each assignment has a single run script to execute all experiments:

```bash
# Assignment 1
cd assignment-1/
bash run_ablation_studies.sh

# Assignment 2
cd assignment-2/
bash run_all.sh
```

---

## Visualization

After running experiments, results are saved in `data/` and `plots/`. Use the included plotting scripts to regenerate all comparison graphs:

```bash
python plot_all.py           # For CartPole
python plot_all_bonus.py     # For Acrobot
```

---

## Reports

Each assignment includes a final PDF write-up following the ICML LaTeX template. Reports include:
- Background theory
- Implementation details
- Experiment setup
- Results and analysis
- Visualizations

---

## Resources

- [Gymnasium (Farama)](https://gymnasium.farama.org/)
- [Sutton & Barto – RL Book](http://incompleteideas.net/book/the-book.html)
- [Spinning Up – OpenAI](https://spinningup.openai.com/en/latest/)

---

## Contact

Feel free to reach out if you'd like to discuss implementation details or results. This repository is maintained as part of my MSc Data Science coursework.

