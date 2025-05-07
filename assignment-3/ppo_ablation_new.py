import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import List, Dict, Any

# --- General Configuration ---
TOTAL_TIMESTEPS = 1_000_000 # Total steps for each training run
BASE_SEED = 42
ENV_NAME = "CartPole-v1"
REPEATS_PER_SETTING = 1 # Number of runs for each hyperparameter setting for averaging
DATA_DIR = "data"
PLOTS_DIR = "plots"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- PPO Default Hyperparameters (can be overridden for specific ablations) ---
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_GAMMA = 0.99
DEFAULT_CLIP_EPS = 0.2
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_STEPS_PER_UPDATE = 2048
DEFAULT_HIDDEN_SIZE = 64 # Base hidden size, varied in its ablation
DEFAULT_ENTROPY_COEFF = 0.01
DEFAULT_VALUE_COEFF = 0.5
DEFAULT_GAE_LAMBDA = 0.95

# --- PPO Core Components ---
class ActorCritic(nn.Module):
    """ Actor-Critic Network """
    def __init__(self, obs_dim: int, n_actions: int, hidden_size_actor_critic: int):
        super().__init__()
        # Shared layers can be considered if desired, but current user code has separate paths after initial shared
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size_actor_critic),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size_actor_critic, hidden_size_actor_critic), nn.Tanh(),
            nn.Linear(hidden_size_actor_critic, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size_actor_critic, hidden_size_actor_critic), nn.Tanh(),
            nn.Linear(hidden_size_actor_critic, 1)
        )

    def forward(self, x: torch.Tensor):
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        state_values = self.critic(shared_features).squeeze(-1)
        return action_probs, state_values

class RolloutBuffer:
    """ Stores transitions collected from the environment """
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def clear(self):
        self.__init__() # Re-initialize all lists

def compute_returns_and_advantages(rewards: List[float], dones: List[bool], values: List[float], gamma: float, lam: float):
    """ Computes GAE returns and advantages """
    returns_list, advantages_list = [], []
    gae = 0.0
    next_value = 0.0 # Value of the next state, 0 if terminal
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - float(dones[i])) - values[i]
        gae = delta + gamma * lam * (1 - float(dones[i])) * gae
        advantages_list.insert(0, gae)
        returns_list.insert(0, gae + values[i]) # Return is advantage + value
        next_value = values[i]
    return torch.tensor(returns_list, dtype=torch.float32), torch.tensor(advantages_list, dtype=torch.float32)

def ppo_update(model: ActorCritic, optimizer: optim.Optimizer, buffer: RolloutBuffer,
               clip_eps_update: float, epochs_update: int, batch_size_update: int,
               value_coeff_update: float, entropy_coeff_update: float, device: torch.device):
    """ Performs PPO updates over collected trajectories """
    # Prepare data from buffer
    states_tensor = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in buffer.states]).to(device)
    actions_tensor = torch.tensor(buffer.actions, dtype=torch.int64).to(device)
    old_log_probs_tensor = torch.tensor(buffer.log_probs, dtype=torch.float32).to(device)

    # Compute returns and advantages
    returns, advantages = compute_returns_and_advantages(
        buffer.rewards, buffer.dones, buffer.values, DEFAULT_GAMMA, DEFAULT_GAE_LAMBDA
    )
    returns = returns.to(device)
    advantages = advantages.to(device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs_update):
        # Shuffle indices for minibatch creation
        indices = np.arange(len(states_tensor))
        np.random.shuffle(indices)
        for i in range(0, len(states_tensor), batch_size_update):
            batch_indices = indices[i : i + batch_size_update]

            # Get data for the current minibatch
            batch_states = states_tensor[batch_indices]
            batch_actions = actions_tensor[batch_indices]
            batch_old_log_probs = old_log_probs_tensor[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            # Get new action probabilities and values from the model
            logits, current_values = model(batch_states)
            distribution = Categorical(logits)
            
            entropy = distribution.entropy().mean()
            new_log_probs = distribution.log_prob(batch_actions)

            # Calculate ratio for PPO objective
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            # Clipped surrogate objective
            surrogate1 = ratio * batch_advantages
            surrogate2 = torch.clamp(ratio, 1 - clip_eps_update, 1 + clip_eps_update) * batch_advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # Value function loss
            critic_loss = (batch_returns - current_values).pow(2).mean()
            
            # Total loss
            loss = actor_loss + value_coeff_update * critic_loss - entropy_coeff_update * entropy

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Optional: gradient clipping
            optimizer.step()

# --- Training Function ---
def run_ppo_training_session(
    run_id: int,
    env_name_train: str,
    total_timesteps_train: int,
    seed_train: int,
    # Hyperparameters to use for this session
    hidden_size_train: int,
    clip_eps_train: float,
    learning_rate_train: float,
    gamma_train: float,
    epochs_train: int,
    batch_size_train: int,
    steps_per_update_train: int,
    gae_lambda_train: float,
    value_coeff_train: float,
    entropy_coeff_train: float,
    # Filename for saving results
    output_csv_filename: str):
    """ Runs a single PPO training session and saves results. """

    print(f"\n--- Starting Training: Run ID {run_id}, Seed {seed_train} ---")
    print(f"Parameters: Hidden Size={hidden_size_train}, Clip Eps={clip_eps_train}, LR={learning_rate_train}")
    print(f"Saving results to: {output_csv_filename}")

    # Set seeds for reproducibility for this specific run
    torch.manual_seed(seed_train)
    np.random.seed(seed_train)

    env = gym.make(env_name_train)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ActorCritic(obs_dim, n_actions, hidden_size_train).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_train)
    rollout_buffer = RolloutBuffer()

    # Logging
    episode_rewards_log: List[float] = []
    total_steps_log: List[int] = []
    
    current_total_env_steps = 0
    episode_count = 0
    
    obs, _ = env.reset(seed=seed_train) # Set seed for environment reset

    while current_total_env_steps < total_timesteps_train:
        # --- Rollout Phase ---
        for _ in range(steps_per_update_train):
            state_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dim
            
            with torch.no_grad():
                action_probs, state_value = model(state_tensor)
            
            distribution = Categorical(action_probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # Store transition in buffer
            rollout_buffer.states.append(obs) # Store as numpy array
            rollout_buffer.actions.append(action.item())
            rollout_buffer.log_probs.append(log_prob.item())
            rollout_buffer.rewards.append(float(reward))
            rollout_buffer.dones.append(done)
            rollout_buffer.values.append(state_value.item())
            
            obs = next_obs
            current_total_env_steps += 1

            if done:
                # Calculate total reward for the episode
                # The current_episode_reward calculation needs to be done carefully if STEPS_PER_UPDATE is long
                # For simplicity, we log rewards when an episode ends within the rollout
                # Find where the episode started in the current rollout buffer for accurate reward sum
                current_episode_reward_sum = 0
                # This logic is tricky if episodes span across ppo_update calls.
                # A simpler way is to track reward sum from env.reset()
                # For now, let's assume we log at the end of the episode based on external tracking
                # The user's original code had a simpler current_episode_reward tracker. Let's use that.
                
                # Simplified episode reward tracking (as in user's original snippet)
                # This part is tricky if an episode doesn't end within STEPS_PER_UPDATE.
                # The provided PPO code logs reward when `done or truncated` happens.
                # We need to find the actual episode rewards.
                # Let's refine this part. We need to sum rewards from the start of the episode.
                
                # To get accurate episode rewards, we need to sum them up from the last reset.
                # This is complex if episodes span multiple rollouts.
                # The original code snippet implies a simpler per-step accumulation.
                # Let's assume the provided `episode_returns` in the original snippet was accurate.
                # We will simulate that by finding the last `done` in the buffer.
                
                # The most straightforward way is to track episode reward outside the inner loop:
                # Initialize current_episode_reward = 0 when obs, _ = env.reset()
                # Add reward to it in the loop.
                # When done, log it and reset.
                # This is what the user's original code did.
                # The current structure collects a fixed number of steps (STEPS_PER_UPDATE)
                # then updates. An episode might not end within this.
                # For accurate logging, we need to track episodes across these collection phases.

                # The user's original code snippet had this logic:
                # current_episode_reward = 0 (after reset)
                # while total_env_steps < TOTAL_TIMESTEPS:
                #   for _ in range(STEPS_PER_UPDATE):
                #     ...
                #     current_episode_reward += reward
                #     if done or truncated:
                #         episode_returns.append(current_episode_reward)
                #         total_steps_list.append(total_env_steps)
                #         current_episode_reward = 0
                #         obs, _ = env.reset()
                # This means we need to adapt the logging.
                # The current `run_ppo_training_session` is designed for one full training.
                # The logging of episode rewards should happen within the `while current_total_env_steps < total_timesteps_train:` loop.

                # Let's adjust the logging part to be similar to the user's original script.
                # This means we need an outer loop for episodes and an inner for steps.
                # The current structure is: outer loop for total steps, inner for STEPS_PER_UPDATE.
                # This is typical for PPO. Logging episode rewards accurately requires care.

                # Let's stick to the PPO structure and log rewards from completed episodes within the rollout.
                # This might mean some episodes are partially completed when an update happens.
                # The most common practice is to log when an episode *actually* finishes.
                pass # Handled by the outer loop's episode tracking logic

            if current_total_env_steps >= total_timesteps_train:
                break
        
        # --- PPO Update Phase ---
        if len(rollout_buffer.states) > 0: # Ensure buffer is not empty
            ppo_update(model, optimizer, rollout_buffer,
                       clip_eps_train, epochs_train, batch_size_train,
                       value_coeff_train, entropy_coeff_train, device)
            rollout_buffer.clear()
        else: # Should not happen if steps_per_update_train > 0
            print("Warning: Rollout buffer was empty before update.")

        # This part is for logging episodes. We need a separate tracking for episode rewards.
        # The original script had a loop that ran until TOTAL_TIMESTEPS,
        # and within that, a loop for STEPS_PER_UPDATE.
        # Episode logging was done when `done or truncated`.

    # Re-introducing episode tracking similar to user's original script for accurate logging
    # This requires restructuring the main loop slightly.
    # The current structure is fine for PPO, but logging needs to be integrated.

    # Let's refine the main loop for proper episode tracking and logging
    # This is a critical part for accurate results.
    
    # --- Main Training Loop (Revised for accurate episode logging) ---
    obs, _ = env.reset(seed=seed_train)
    current_episode_reward_sum = 0.0
    current_total_env_steps_accurate = 0 # Use this for logging steps

    while current_total_env_steps_accurate < total_timesteps_train:
        # Collect data for STEPS_PER_UPDATE or until total_timesteps_train is reached
        for _ in range(steps_per_update_train):
            if current_total_env_steps_accurate >= total_timesteps_train:
                break

            state_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs, state_value = model(state_tensor)
            
            distribution = Categorical(action_probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            rollout_buffer.states.append(obs)
            rollout_buffer.actions.append(action.item())
            rollout_buffer.log_probs.append(log_prob.item())
            rollout_buffer.rewards.append(float(reward))
            rollout_buffer.dones.append(done)
            rollout_buffer.values.append(state_value.item())
            
            obs = next_obs
            current_episode_reward_sum += float(reward)
            current_total_env_steps_accurate += 1

            if done:
                episode_rewards_log.append(current_episode_reward_sum)
                total_steps_log.append(current_total_env_steps_accurate)
                episode_count += 1
                print(f"Run {run_id}, Episode {episode_count}: Reward = {current_episode_reward_sum:.2f}, Total Steps = {current_total_env_steps_accurate}")
                obs, _ = env.reset(seed=seed_train + episode_count) # Vary seed slightly for subsequent episodes
                current_episode_reward_sum = 0.0
        
        # Perform PPO update
        if len(rollout_buffer.states) > 0:
            ppo_update(model, optimizer, rollout_buffer,
                       clip_eps_train, epochs_train, batch_size_train,
                       value_coeff_train, entropy_coeff_train, device)
            rollout_buffer.clear()

    env.close()

    # Save results for this run
    if episode_rewards_log: # Check if any episodes were completed
        run_df = pd.DataFrame({
            "Run": run_id,
            "Episode": range(1, len(episode_rewards_log) + 1),
            "Total Steps": total_steps_log,
            "Episode Reward": episode_rewards_log
        })
        run_df.to_csv(output_csv_filename, index=False)
        print(f"Successfully saved results to {output_csv_filename}")
    else:
        print(f"Warning: No episodes completed for Run ID {run_id}. No CSV saved to {output_csv_filename}")


# --- Plotting Function ---
def plot_ablation_study(
    param_name_to_ablate: str, # e.g., "Hidden Size" or "Clip Epsilon"
    param_values_tested: List[Any],
    file_pattern_template: str, # e.g., "data/ppo_ablation_hidden_H{}_defaultClip_returns.csv"
                                # or "data/ppo_ablation_clip_C{}_defaultHidden_returns.csv"
    plot_title_main: str,
    output_plot_filename_path: str,
    xlabel_plot: str = "Total Environment Steps",
    ylabel_plot: str = "Average Episode Reward",
    downsample_factor_plot: int = 100,
    smoothing_window_plot: int = 200
    ):
    """ Generates and saves a plot for an ablation study. """
    print(f"\n--- Generating Plot: {plot_title_main} ---")
    print(f"Saving plot to: {output_plot_filename_path}")

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.labelsize': 14, 'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 11, 'legend.title_fontsize': 13, 'lines.linewidth': 2.0,
        'figure.dpi': 100, 'savefig.dpi': 300, 'axes.edgecolor': 'gray',
        'grid.color': 'lightgray', 'grid.linestyle': '--', 'grid.linewidth': 0.7,
    })

    plt.figure(figsize=(12, 7))
    all_max_steps = 0

    for i, param_value in enumerate(param_values_tested):
        # Construct filename based on the parameter value
        # Need to handle float formatting for filenames carefully (e.g. clip_eps 0.1 -> "0_1")
        if isinstance(param_value, float):
            param_value_str = str(param_value).replace('.', '_')
        else:
            param_value_str = str(param_value)

        filepath = file_pattern_template.format(param_value_str)

        if not os.path.exists(filepath):
            print(f"Plotting: Skipping {param_name_to_ablate} = {param_value}. File not found: {filepath}")
            continue
        
        try:
            df = pd.read_csv(filepath)
            if "Total Steps" not in df.columns or "Episode Reward" not in df.columns:
                print(f"Plotting: Skipping {param_name_to_ablate} = {param_value}. Missing required columns in {filepath}.")
                continue
            if df.empty:
                print(f"Plotting: Skipping {param_name_to_ablate} = {param_value}. File is empty: {filepath}")
                continue
        except pd.errors.EmptyDataError:
            print(f"Plotting: Skipping {param_name_to_ablate} = {param_value}. File is empty: {filepath}")
            continue
        except Exception as e:
            print(f"Plotting: Skipping {param_name_to_ablate} = {param_value}. Error reading {filepath}: {e}")
            continue
        
        # Aggregate results if multiple runs are in the same CSV (Run column exists)
        # For now, assuming each CSV is one run or already aggregated if REPEATS_PER_SETTING > 1
        # The provided training script saves one CSV per (setting, run_id).
        # So, if REPEATS_PER_SETTING > 1, we need to load and average them.
        # For simplicity, this plotting function assumes the CSV is ready for plotting (one line per episode).
        # If REPEATS_PER_SETTING > 1, the calling code should handle loading multiple CSVs for that setting.
        
        # The current setup: run_ppo_training_session saves one CSV per run_id.
        # So, if REPEATS_PER_SETTING > 1, we'd have multiple files for the same (param_value).
        # This plotting function needs to be aware of that.

        # Let's assume for now that file_pattern_template points to a single CSV
        # that might contain multiple runs if they were concatenated, or just one run.
        # The groupby handles this if a 'Run' column is present.
        # If each file is one run, we need to load all files for a given param_value.

        # Simpler approach: Assume each CSV is for ONE setting and ONE run.
        # If REPEATS_PER_SETTING > 1, the main script will generate multiple files like:
        # ppo_ablation_hidden_H64_run0_returns.csv
        # ppo_ablation_hidden_H64_run1_returns.csv
        # The plotting function will need to load all these for H64, then average.

        # --- Revised data loading for multiple runs per setting ---
        all_dfs_for_param_value = []
        for run_idx in range(REPEATS_PER_SETTING):
            # Adjust file_pattern_template to include run_idx if REPEATS_PER_SETTING > 1
            # Current file_pattern_template does not have a placeholder for run_idx.
            # Let's assume the filename passed to run_ppo_training_session already includes run_idx.
            # And file_pattern_template here is for a specific run.
            # This means the caller of plot_ablation_study needs to handle averaging across runs.

        # For now, let's stick to the original plotting script's logic:
        # it processes one CSV at a time, assuming that CSV contains all data for that line on the plot.
        # If REPEATS_PER_SETTING > 1, the run_ppo_training_session saves separate files.
        # This means the `file_pattern_template` needs to be more specific or this function needs to loop runs.

        # Simplification: Plotting function assumes each CSV is one line on the graph.
        # If averaging over runs is needed, it must be done *before* calling this,
        # or this function must be modified to load multiple run CSVs for each param_value.
        # The user's original plot script processes one CSV per line. Let's keep that.
        # This means if REPEATS_PER_SETTING > 1, the main script should first aggregate those into one CSV per param_value.
        # OR, the training script saves an aggregated CSV if REPEATS_PER_SETTING > 1.
        # The current `run_ppo_training_session` saves one CSV per run.

        # Let's modify the plotting to average if multiple files for the same param_value exist (due to REPEATS_PER_SETTING)
        # This requires a change in how `file_pattern_template` is used.
        # Assume `file_pattern_template` is like "data/ppo_ablation_hidden_H{}_clip_default_run{}"
        
        # Sticking to the user's original plotting script style: one CSV per plotted line.
        # This implies that if REPEATS_PER_SETTING > 1, the training part should produce an *aggregated* CSV
        # for each `param_value`, or the main script should aggregate them before plotting.
        # The current training script saves `ppo_ablation_X_Y_runZ.csv`.

        # Let's make the plotting function load all run files for a given `param_value`.
        dfs_for_current_param = []
        for r_idx in range(REPEATS_PER_SETTING):
            # file_pattern_template should have two placeholders: one for param_value, one for run_idx
            # e.g. "data/ablation_hidden_H{}_run{}.csv"
            # The current file_pattern_template has one placeholder.
            # This means the caller needs to provide a list of files for each param_value if REPEATS_PER_SETTING > 1.

        # Simplest for now: Assume the CSV at `filepath` is what we plot.
        # If REPEATS_PER_SETTING > 1, the main script needs to ensure `filepath` points to an aggregated CSV,
        # or the `run_ppo_training_session` saves an aggregated one.
        # The current `run_ppo_training_session` saves one CSV per run.

        # OK, the most robust way: `plot_ablation_study` takes a list of filepaths for each `param_value` if REPEATS_PER_SETTING > 1.
        # Or, it constructs them. Let's assume it constructs them if REPEATS_PER_SETTING > 1.

        # --- Data Aggregation for Multiple Runs (if REPEATS_PER_SETTING > 1) ---
        # This part needs the `file_pattern_template` to accept `param_value_str` and a `run_idx`.
        # Let's assume `file_pattern_template` is now e.g. "data/ppo_H{}_run{}.csv"
        # And `param_value_str` is inserted by the caller.
        # This function will get a base_filepath like "data/ppo_H{param_value_str}"
        # And append "_run{run_idx}.csv"

        # This is getting complicated. Let's revert to the simpler model:
        # The main script is responsible for providing the correct (possibly aggregated) CSV path.
        # The plot function plots data from ONE such CSV per `param_value`.
        # If REPEATS_PER_SETTING > 1, the main script should aggregate first or the training should save aggregated.
        # The user's original plotting script takes one CSV per line.

        # The current training script saves one CSV per run.
        # So, if REPEATS_PER_SETTING > 1, we need to load all of them for a given setting.
        
        all_rewards_for_setting: List[pd.Series] = []
        max_steps_for_setting = 0

        for run_iter in range(REPEATS_PER_SETTING):
            # Construct filepath for each run
            # This requires file_pattern_template to be like "data/ppo_H{}_clip_default_run{}.csv"
            # The current passed `filepath` is "data/ppo_ablation_hidden_H{param_value_str}_defaultClip_returns.csv"
            # This implies REPEATS_PER_SETTING = 1 or aggregation happened before.
            
            # Let's assume `filepath` is the single file to plot for this `param_value`.
            # If REPEATS_PER_SETTING > 1, this means the main script must have created an aggregated file
            # or the training function saved an aggregated file.
            # The current training function saves one file per run.
            # So, this plotting function needs to load multiple files if REPEATS_PER_SETTING > 1.

            # --- MODIFIED DATA LOADING FOR MULTIPLE RUNS ---
            # The `file_pattern_template` should be the base, and we append `_run{run_idx}`
            # Example: `file_pattern_template` = "data/ppo_H{}_clip_default"
            # `param_value_str` = "64"
            # Then loop `run_idx` and load `file_pattern_template.format(param_value_str) + f"_run{run_idx}_returns.csv"`
            
            # Let's make `file_pattern_template` the full name for a single run,
            # and the loop for `param_values_tested` will iterate through settings.
            # If REPEATS_PER_SETTING > 1, we need to collect all DFs for that setting.
            
            # New approach: The main script will prepare a list of DFs for each param_value if REPEATS_PER_SETTING > 1
            # This function will then average them.
            # For now, assume df is loaded from a single filepath.
            
            if "Total Steps" not in df.columns or df["Total Steps"].empty:
                 print(f"Warning: 'Total Steps' is empty or missing for {filepath}. Skipping this file for this param value.")
                 continue
            
            # Group by 'Total Steps' for consistent x-axis, then average 'Episode Reward'
            # This is important if steps are not perfectly aligned across runs/episodes
            # However, usually, we align by episode number or smooth raw rewards.
            # The original script groups by Total Steps.
            
            # For multiple runs, we need to align them. A common way is to interpolate.
            # Or, if all runs have similar step counts, we can average rewards at similar step intervals.
            # The simplest is to average rewards for episodes that end within certain step bins.
            
            # Let's use the logic from the user's original plotting script for smoothing and plotting one line.
            # This means if REPEATS_PER_SETTING > 1, the data in `df` should already be from multiple runs
            # (e.g., concatenated by the training script).
            # The current training script does `final_df = pd.concat(all_runs, ignore_index=True)` if REPEATS_PER_SETTING > 1
            # *within one call* to `run_ppo_training_session`.
            # So, the CSV saved by `run_ppo_training_session` will contain all repeats for that setting. Good.

            grouped = df.groupby("Total Steps")["Episode Reward"] # Group by steps
            mean_rewards_at_steps = grouped.mean()
            std_rewards_at_steps = grouped.std().fillna(0)
            
            if mean_rewards_at_steps.empty:
                print(f"Warning: No data after grouping for {filepath}. Skipping.")
                continue

            steps_raw = mean_rewards_at_steps.index
            
            current_smoothing_window = min(smoothing_window_plot, len(mean_rewards_at_steps))
            if current_smoothing_window < 1: current_smoothing_window = 1
            
            smoothed_means = mean_rewards_at_steps.rolling(window=current_smoothing_window, min_periods=1, center=True).mean()
            smoothed_stds = std_rewards_at_steps.rolling(window=current_smoothing_window, min_periods=1, center=True).mean()

            # Downsample
            steps_plot = steps_raw[::downsample_factor_plot]
            smoothed_means_plot = smoothed_means.reindex(steps_plot, method='nearest').fillna(method='ffill').fillna(method='bfill') # Ensure correct indexing
            smoothed_stds_plot = smoothed_stds.reindex(steps_plot, method='nearest').fillna(method='ffill').fillna(method='bfill')
            
            if steps_plot.empty or smoothed_means_plot.empty:
                print(f"Warning: No data after downsampling for {filepath}. Skipping.")
                continue
                
            all_max_steps = max(all_max_steps, steps_plot.max() if not steps_plot.empty else 0)

            plt.plot(steps_plot, smoothed_means_plot, label=f"{param_name_to_ablate} = {param_value}")
            plt.fill_between(
                steps_plot,
                (smoothed_means_plot - smoothed_stds_plot).clip(lower=0),
                (smoothed_means_plot + smoothed_stds_plot),
                alpha=0.15
            )

    plt.xlabel(xlabel_plot)
    plt.ylabel(ylabel_plot)
    plt.title(plot_title_main, fontweight='bold')
    
    # Add reference lines for CartPole
    if "CartPole" in plot_title_main:
        plt.axhline(y=500, color='grey', linestyle=':', linewidth=1.5, label='Max Score (500)')
        plt.axhline(y=475, color='darkgrey', linestyle=':', linewidth=1.5, label='Solved Threshold (475)')

    legend = plt.legend(title=param_name_to_ablate, loc="lower right", frameon=True, fancybox=True)
    if legend: legend.get_frame().set_edgecolor('dimgray')

    plt.xlim(left=0, right=all_max_steps if all_max_steps > 0 else TOTAL_TIMESTEPS) # Adjust xlim
    plt.ylim(bottom=0, top=550 if "CartPole" in plot_title_main else None) # Adjust ylim for CartPole

    def k_formatter(x, pos): return f'{int(x/1000)}k' if x >= 1000 else int(x)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth='0.6', alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', linewidth='0.4', alpha=0.5)
    plt.tight_layout(pad=1.5)
    plt.savefig(output_plot_filename_path)
    plt.show()
    print(f"Plot successfully saved to {output_plot_filename_path}")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Ablation Study Definitions ---
    hidden_sizes_to_test = [64, 128, 256, 512]
    clip_eps_values_to_test = [0.1, 0.2, 0.3] # Example values

    fixed_clip_eps_for_hidden_ablation = DEFAULT_CLIP_EPS
    # Choose a good hidden size for clip_eps ablation (e.g., from hidden ablation results or default)
    fixed_hidden_size_for_clip_ablation = 128 # Example: chosen after hidden_size ablation

    # --- Phase 1: Training ---
    print("===== Starting Training Phase =====")

    # 1. Hidden Size Ablation Study
    print("\n--- Running Hidden Size Ablation Study ---")
    for i, hs_value in enumerate(hidden_sizes_to_test):
        for r_idx in range(REPEATS_PER_SETTING): # Loop for repeats
            current_seed = BASE_SEED + r_idx
            # Filename includes all key varying and fixed parameters for clarity
            # If REPEATS_PER_SETTING > 1, the plot function needs to know how to find all run files for a setting.
            # Let's make filenames include the run index.
            output_filename = os.path.join(DATA_DIR, f"ppo_ablation_hidden_H{hs_value}_clip{str(fixed_clip_eps_for_hidden_ablation).replace('.', '_')}_run{r_idx}.csv")
            
            # The run_ppo_training_session now saves one CSV per call (per run_id).
            # If REPEATS_PER_SETTING > 1, multiple CSVs will be generated for each `hs_value`.
            # The plotting function will need to load and average these.
            # For now, let's assume the training function saves an *aggregated* CSV if REPEATS_PER_SETTING > 1
            # The current `run_ppo_training_session` saves one CSV per run.
            # The plotting function needs to be adapted.

            # Let's adjust run_ppo_training_session to save one *aggregated* CSV if REPEATS_PER_SETTING > 1.
            # No, the current `run_ppo_training_session` takes `run_id` and saves one file.
            # The main loop here calls it `REPEATS_PER_SETTING` times.
            # The plotting function has been updated to expect this.
            
            # The `output_csv_filename` for `run_ppo_training_session` should be unique per run.
            # The `file_pattern_template` for `plot_ablation_study` will then use this.
            
            # Let's make the filename for run_ppo_training_session specific to that run.
            # The plotting function will then need a way to find all relevant run files.

            # Simpler: The filename passed to run_ppo_training_session is the final one for that run.
            # The plotting function will then be given a list of these filenames if REPEATS_PER_SETTING > 1.
            # Or, plot_ablation_study constructs the filenames.

            # Let's stick to: `run_ppo_training_session` saves one file per run.
            # `plot_ablation_study` will be given a file *pattern* and iterate through runs.
            # This means `file_pattern_template` in `plot_ablation_study` needs a run placeholder.
            # E.g., "data/ppo_ablation_hidden_H{param_val}_clip_default_run{run_idx}.csv"

            # The current `plot_ablation_study` expects `file_pattern_template.format(param_value_str)` to be the path.
            # This works if REPEATS_PER_SETTING = 1.
            # If REPEATS_PER_SETTING > 1, the `run_ppo_training_session` saves multiple files.
            # The plotting function needs to load and average these.

            # Let's make the training loop save one CSV per (setting, run_idx)
            # And the plotting function will load all CSVs for a given setting and average them.
            
            # Filename for individual run:
            run_specific_csv_path = os.path.join(DATA_DIR, f"ppo_H{hs_value}_clip{str(fixed_clip_eps_for_hidden_ablation).replace('.', '_')}_run{r_idx}_temp.csv")

            run_ppo_training_session(
                run_id=r_idx, env_name_train=ENV_NAME, total_timesteps_train=TOTAL_TIMESTEPS, seed_train=current_seed,
                hidden_size_train=hs_value, clip_eps_train=fixed_clip_eps_for_hidden_ablation,
                learning_rate_train=DEFAULT_LEARNING_RATE, gamma_train=DEFAULT_GAMMA,
                epochs_train=DEFAULT_EPOCHS, batch_size_train=DEFAULT_BATCH_SIZE,
                steps_per_update_train=DEFAULT_STEPS_PER_UPDATE, gae_lambda_train=DEFAULT_GAE_LAMBDA,
                value_coeff_train=DEFAULT_VALUE_COEFF, entropy_coeff_train=DEFAULT_ENTROPY_COEFF,
                output_csv_filename=run_specific_csv_path
            )
        # After all runs for a given hs_value, aggregate them into one CSV for plotting
        all_run_dfs_hs = []
        for r_idx in range(REPEATS_PER_SETTING):
            temp_f_path = os.path.join(DATA_DIR, f"ppo_H{hs_value}_clip{str(fixed_clip_eps_for_hidden_ablation).replace('.', '_')}_run{r_idx}_temp.csv")
            if os.path.exists(temp_f_path):
                all_run_dfs_hs.append(pd.read_csv(temp_f_path))
        if all_run_dfs_hs:
            aggregated_df_hs = pd.concat(all_run_dfs_hs, ignore_index=True)
            aggregated_csv_path_hs = os.path.join(DATA_DIR, f"ppo_ablation_hidden_H{hs_value}_clip{str(fixed_clip_eps_for_hidden_ablation).replace('.', '_')}.csv")
            aggregated_df_hs.to_csv(aggregated_csv_path_hs, index=False)
            print(f"Aggregated CSV for Hidden Size {hs_value} saved to {aggregated_csv_path_hs}")
             # Clean up temporary run files
            for r_idx in range(REPEATS_PER_SETTING):
                 temp_f_path = os.path.join(DATA_DIR, f"ppo_H{hs_value}_clip{str(fixed_clip_eps_for_hidden_ablation).replace('.', '_')}_run{r_idx}_temp.csv")
                 if os.path.exists(temp_f_path): os.remove(temp_f_path)


    # 2. Clip Epsilon Ablation Study
    print("\n--- Running Clip Epsilon Ablation Study ---")
    for i, ce_value in enumerate(clip_eps_values_to_test):
        for r_idx in range(REPEATS_PER_SETTING):
            current_seed = BASE_SEED + r_idx + REPEATS_PER_SETTING # Offset seed
            run_specific_csv_path_ce = os.path.join(DATA_DIR, f"ppo_clip{str(ce_value).replace('.', '_')}_hidden{fixed_hidden_size_for_clip_ablation}_run{r_idx}_temp.csv")
            run_ppo_training_session(
                run_id=r_idx, env_name_train=ENV_NAME, total_timesteps_train=TOTAL_TIMESTEPS, seed_train=current_seed,
                hidden_size_train=fixed_hidden_size_for_clip_ablation, clip_eps_train=ce_value,
                learning_rate_train=DEFAULT_LEARNING_RATE, gamma_train=DEFAULT_GAMMA,
                epochs_train=DEFAULT_EPOCHS, batch_size_train=DEFAULT_BATCH_SIZE,
                steps_per_update_train=DEFAULT_STEPS_PER_UPDATE, gae_lambda_train=DEFAULT_GAE_LAMBDA,
                value_coeff_train=DEFAULT_VALUE_COEFF, entropy_coeff_train=DEFAULT_ENTROPY_COEFF,
                output_csv_filename=run_specific_csv_path_ce
            )
        # Aggregate for clip_eps
        all_run_dfs_ce = []
        for r_idx in range(REPEATS_PER_SETTING):
            temp_f_path = os.path.join(DATA_DIR, f"ppo_clip{str(ce_value).replace('.', '_')}_hidden{fixed_hidden_size_for_clip_ablation}_run{r_idx}_temp.csv")
            if os.path.exists(temp_f_path):
                all_run_dfs_ce.append(pd.read_csv(temp_f_path))
        if all_run_dfs_ce:
            aggregated_df_ce = pd.concat(all_run_dfs_ce, ignore_index=True)
            aggregated_csv_path_ce = os.path.join(DATA_DIR, f"ppo_ablation_clip_C{str(ce_value).replace('.', '_')}_hidden{fixed_hidden_size_for_clip_ablation}.csv")
            aggregated_df_ce.to_csv(aggregated_csv_path_ce, index=False)
            print(f"Aggregated CSV for Clip Eps {ce_value} saved to {aggregated_csv_path_ce}")
            for r_idx in range(REPEATS_PER_SETTING): # Clean up
                 temp_f_path = os.path.join(DATA_DIR, f"ppo_clip{str(ce_value).replace('.', '_')}_hidden{fixed_hidden_size_for_clip_ablation}_run{r_idx}_temp.csv")
                 if os.path.exists(temp_f_path): os.remove(temp_f_path)


    # --- Phase 2: Plotting ---
    print("\n===== Starting Plotting Phase =====")

    # 1. Plot Hidden Size Ablation
    plot_ablation_study(
        param_name_to_ablate="Hidden Size",
        param_values_tested=hidden_sizes_to_test,
        file_pattern_template=os.path.join(DATA_DIR, f"ppo_ablation_hidden_H{{}}_clip{str(fixed_clip_eps_for_hidden_ablation).replace('.', '_')}.csv"),
        plot_title_main=f"PPO Hidden Layer Ablation (Clip $\epsilon$={fixed_clip_eps_for_hidden_ablation}) on {ENV_NAME}",
        output_plot_filename_path=os.path.join(PLOTS_DIR, "ppo_ablation_hidden_size.png")
    )

    # 2. Plot Clip Epsilon Ablation
    plot_ablation_study(
        param_name_to_ablate="Clip Epsilon ($\epsilon$)",
        param_values_tested=clip_eps_values_to_test,
        file_pattern_template=os.path.join(DATA_DIR, f"ppo_ablation_clip_C{{}}_hidden{fixed_hidden_size_for_clip_ablation}.csv"),
        plot_title_main=f"PPO Clip Epsilon Ablation (Hidden Size={fixed_hidden_size_for_clip_ablation}) on {ENV_NAME}",
        output_plot_filename_path=os.path.join(PLOTS_DIR, "ppo_ablation_clip_epsilon.png")
    )

    print("\nAll experiments and plotting completed.")

