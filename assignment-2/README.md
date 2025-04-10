# Assignment 2: Research Study on Policy Based RL Methods

## Reproducing Policy Gradient Experiments on DSLab/Computer Lab Machines

This repository provides reproducible implementations for comparing REINFORCE, Actor-Critic (AC), and Advantage Actor-Critic (A2C) on the CartPole-v1 and Acrobot-v1 environments.

---

## 1. Environment Setup (one-time only)
Make sure Python and pip are installed, then install the necessary dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Directory Structure
Your working directory should look like this:

```
.
├── data/                  # Output CSVs saved here
├── plots/                 # Visualizations will be saved here
├── models.py              # Shared Actor-Critic model definition
├── reinforce.py           # REINFORCE on CartPole
├── actor_critic.py        # AC on CartPole
├── a2c.py                 # A2C on CartPole
├── reinforce_acrobot.py   # REINFORCE on Acrobot
├── actor_critic_acrobot.py
├── a2c_acrobot.py         # A2C on Acrobot
├── plot_pg_comparison.py  # Plotting script for policy gradient methods
```

---

## 3. Run Experiments (Single Command per Task)
### CartPole-v1

```bash
python reinforce.py
python actor_critic.py
python a2c.py
```

### Acrobot-v1

```bash
python reinforce_acrobot.py
python actor_critic_acrobot.py
python a2c_acrobot.py
```

---

## 4. Visualize Results
To generate a comparison plot after all experiments are completed:

```bash
python plot_all.py # Cartpole Results
python plot_bonus.py # Acrobot Results
```
This will save a plot at `plots/policy_gradient_comparison.png`

---

## 5. Notes
- All experiment logs are stored as `.csv` files in the `data/` folder.
- Each run logs episodic return and total steps.
- Experiments are reproducible and deterministic with the specified seed.


