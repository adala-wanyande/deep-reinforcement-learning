# Running the Experiments

This document provides a step-by-step guide on how to manually execute the experiments for **Naive DQN** and **Stable DQN** ablation studies. Follow the instructions below to ensure the experiments run correctly and results are saved properly.

---

## 1. Setup Environment

Ensure you have the necessary dependencies installed. Run the following command to install the required packages:

```bash
pip install -r requirements.txt  # If a requirements file exists
```

If no `requirements.txt` file is available, manually install key dependencies:

```bash
pip install torch gymnasium numpy pandas matplotlib seaborn
```

---

## 2. Running the Naive DQN Experiments

Navigate to the directory containing the Python scripts and execute the following commands **in order**:

### **Ablation Studies for Naive DQN**

```bash
python3 ablation_learning_rate.py  # Test impact of learning rate
python3 ablation_discount.py  # Test impact of discount factor (gamma)
python3 ablation_exploration.py  # Test impact of exploration rate (epsilon decay)
python3 ablation_nn_size.py  # Test impact of neural network size
```

### **Running Naive DQN with Best Hyperparameters**

After identifying the best hyperparameters from the ablation studies, execute:

```bash
python3 best_hyperparameters.py
```

This will run the Naive DQN with the optimal hyperparameters and store results for comparison with the stable versions.

---

## 3. Running the Stable DQN Experiments

### **Ablation Studies for Stabilization Techniques**

Execute the following scripts to test the impact of different stabilization techniques:

```bash
python3 ablation_experience_replay.py  # Test experience replay
python3 ablation_target_network.py  # Test target networks
python3 ablation_both.py  # Test experience replay + target networks
```

---

## 4. Visualizing the Results

After running all experiments, generate plots using:

```bash
python3 plot_all.py  # Generates all comparison plots
```


## 5. Interpreting the Results

- Results for each experiment will be stored in the `data/` directory.
- Generated plots will be saved in the `visuals/` directory.
- The best hyperparameter configuration for DQN should be compared with the stabilization techniques to evaluate improvements.

---

## 6. Notes

- Ensure all experiments are run from the same directory where the scripts are located.
- If any script fails, check dependencies and error messages.
- The ablation studies should be run before training with best hyperparameters to ensure proper comparisons.

This manual approach ensures complete transparency in running and reproducing the experiments. If any issues arise, verify that dependencies are correctly installed and that all scripts are executed sequentially.

The full code for this assignment can be found on my Github: https://github.com/adala-wanyande/deep-rl/tree/main/assignment-1 