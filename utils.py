import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from configs import DEFAULT_MODELS_PATH, DEFAULT_OUTPUTS_PATH
from agents import Agent


def plot_and_compare_train_episode_duration(
        json_file_1: str,
        json_file_2: str,
        window: int,
        path: str = None,
    ):
    """
    Compare the training episode durations of two agents whose statistics are stored in json files.
    """ 

    path = DEFAULT_OUTPUTS_PATH if path is None else path
    if not os.path.isfile(os.path.join(path, json_file_1)):
        raise FileNotFoundError(f"File {json_file_1} not found at: {path}")
    if not os.path.isfile(os.path.join(path, json_file_2)):
        raise FileNotFoundError(f"File {json_file_2} not found at: {path}")
    
    with open(os.path.join(path, json_file_1), "r") as f:
        agent1_dict = json.load(f)
    
    with open(os.path.join(path, json_file_2), "r") as f:
        agent2_dict = json.load(f)

    if len(agent1_dict["train_episode_durations"]) != len(agent2_dict["train_episode_durations"]):
        raise ValueError("Agents have different number of training episodes...")
    
    exp_name_1 = agent1_dict["experiment_name"]
    exp_name_2 = agent2_dict["experiment_name"]
    train_episodes_1 = np.array(agent1_dict["train_episode_durations"])
    train_episodes_2 = np.array(agent2_dict["train_episode_durations"])
    
    if window is not None and window > len(train_episodes_1):
        print("Window size should be smaller than the number of training episodes...")
        return
    
    indexes = np.arange(1, len(train_episodes_1) + 1)
    if window is None: # cumulative average
        avg_episode_dur_1 = np.cumsum(train_episodes_1) / indexes
        avg_episode_dur_2 = np.cumsum(train_episodes_2) / indexes
        title_tag = "(cumulative average)"
    else: # moving average
        incomplete_avg_1 = np.cumsum(train_episodes_1[:window-1]) / np.arange(1, window)
        complete_avg_1 = np.convolve(train_episodes_1, np.ones(window), 'valid') / window
        avg_episode_durations_1 = np.hstack([incomplete_avg_1, complete_avg_1])
        incomplete_avg_2 = np.cumsum(train_episodes_2[:window-1]) / np.arange(1, window)
        complete_avg_2 = np.convolve(train_episodes_2, np.ones(window), 'valid') / window
        avg_episode_durations_2 = np.hstack([incomplete_avg_2, complete_avg_2])
        title_tag = f"(average window = {window})"

    # curve
    plt.plot(indexes, avg_episode_durations_1, label=exp_name_1)
    plt.plot(indexes, avg_episode_durations_2, label=exp_name_2)

    # axis, title and legend
    plt.xlabel("Training episode index")
    plt.ylabel(f"Averaged episode duration")
    plt.title(f"Train episode duration over training {title_tag}")
    plt.legend(loc="upper left")

    # save plot
    plt.savefig(os.path.join(path, f"DIFF_train_{exp_name_1}_{exp_name_2}.png"))
    plt.close()


def plot_and_compare_eval_episode_duration(
        json_file_1: str,
        json_file_2: str,
        window: int,
        path: str = None,
    ):
    """
    Compare the evaluation episode durations of two agents whose statistics are stored in json files.
    """ 

    path = DEFAULT_OUTPUTS_PATH if path is None else path
    if not os.path.isfile(os.path.join(path, json_file_1)):
        raise FileNotFoundError(f"File {json_file_1} not found at: {path}")
    if not os.path.isfile(os.path.join(path, json_file_2)):
        raise FileNotFoundError(f"File {json_file_2} not found at: {path}")
    
    with open(os.path.join(path, json_file_1), "r") as f:
        agent1_dict = json.load(f)
    
    with open(os.path.join(path, json_file_2), "r") as f:
        agent2_dict = json.load(f)
    
    exp_name_1 = agent1_dict["experiment_name"]
    exp_name_2 = agent2_dict["experiment_name"]
    train_ep_idx_1 = np.array(agent1_dict["train_episode_indexes"])
    train_ep_idx_2 = np.array(agent2_dict["train_episode_indexes"])
    eval_episodes_1 = np.array(agent1_dict["eval_episode_durations"])
    eval_episodes_2 = np.array(agent2_dict["eval_episode_durations"])
    max_length_1 = agent1_dict["max_episode_length_eval"]
    max_length_2 = agent2_dict["max_episode_length_eval"]  

    if len(train_ep_idx_1) == 0 or len(eval_episodes_1) == 0:
        raise ValueError("No evaluation episode durations to plot for agent 1...")
    
    if len(train_ep_idx_2) == 0 or len(eval_episodes_2) == 0:
        raise ValueError("No evaluation episode durations to plot for agent 2...")
    
    if len(train_ep_idx_1) != len(train_ep_idx_2):
        raise ValueError("Agents have different number of training episodes...")
    
    if max_length_1 != max_length_2:
        raise ValueError("Agents have different maximum episode lengths...")
    
    # curve
    plt.plot(train_ep_idx_1, [np.mean(lengths) for lengths in eval_episodes_1], label=exp_name_1)
    plt.plot(train_ep_idx_2, [np.mean(lengths) for lengths in eval_episodes_2], label=exp_name_2)

    # axis, title and legend
    plt.xlabel("Training episode index")
    plt.ylabel("Averaged episode duration")
    plt.title(f"Evaluation episode duration over training (limit={max_length_1})")
    plt.legend(loc="upper left")

    # save plot
    plt.savefig(os.path.join(path, f"DIFF_eval_{exp_name_1}_{exp_name_2}.png"))
    plt.close()


def plot_epsilon_scheduler(
        final_epsilon: float = 0.01, 
        n_episodes: int = 2000, 
        path: str = None
    ):
    """
    Plot the epsilon decay over the training episodes.
    """
    path = DEFAULT_OUTPUTS_PATH if path is None else path
    Path(path).mkdir(parents=True, exist_ok=True)

    log_final_epsilon = np.log(final_epsilon)
    epsilon_list = [np.exp(log_final_epsilon * float(i) / n_episodes) for i in range(n_episodes)]

    plt.plot(epsilon_list)
    plt.xlabel("Training episode index")
    plt.ylabel("$\epsilon$")
    plt.title("Exponential epsilon decay over training episodes")
    plt.savefig(os.path.join(path, "epsilon_scheduler.png"))
    plt.close()


def plot_state_value_function(agent_filename: str, path: str = None, save_only: bool = False):
    """
    Plot the state value function of a model stored in a json file.
    """

    agent_filename = agent_filename[:-5] if agent_filename.endswith(".json") else agent_filename # remove extension
    agent = Agent.from_pretrained(agent_filename, path=path)

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    for (x,y) in agent.q_values.keys():
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    x_range = max_x - min_x + 1
    y_range = max_y - min_y + 1
    
    x_values = np.arange(min_x, max_x + 1, step=1)
    y_values = np.arange(min_y, max_y + 1, step=1)

    x_mesh, y_mesh = np.meshgrid(x_values, y_values)

    q_values = np.zeros((y_range, x_range)) # x_mesh.shape == y_mesh.shape
    for (x,y), q in agent.q_values.items():
        q_values[int(y-min_y), int(x-min_x)] = np.max(q)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    # 3D plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surface = ax.plot_surface(x_mesh, y_mesh, q_values, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zticks([])
    ax.set_xlabel("dx")
    ax.set_ylabel("dy")
    fig.colorbar(surface, shrink=0.5, aspect=5)
    plt.title(f"State value function of {agent_filename}")

    if not save_only:
        plt.show()
    else:
        plt.savefig(os.path.join(DEFAULT_OUTPUTS_PATH, f"state_value_function_{agent_filename}.png"))

    plt.close()


def main():

    path = "outputs/"

    plot_and_compare_train_episode_duration(
        json_file_1="mc_train_durations.json",
        json_file_2="sarsa-lambda_train_durations.json",
        window=100,
        path=path,
    )

    plot_and_compare_eval_episode_duration(
        json_file_1="mc_eval_durations.json",
        json_file_2="sarsa-lambda_eval_durations.json",
        window=100,
        path=path,
    )

    plot_epsilon_scheduler(path=path)

    plot_state_value_function(agent_filename="MCAgent")
    plot_state_value_function(agent_filename="SARSALambdaAgent")


if __name__ == "__main__":
    main()