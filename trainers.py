import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
import gymnasium as gym

import text_flappy_bird_gym
from agents import Agent, MCAgent, SARSALambdaAgent
from configs import DEFAULT_OUTPUTS_PATH


class Trainer():

    def __init__(self) -> None:
        self.env = None
        self.agent = None
        self.n_episodes = None
        self.final_epsilon = None
        self.discount_factor = None
        self.experiment_name = None
        self.max_episode_length_eval = None
        self.reset_stats()
    
    def __str__(self) -> str:
        return self.__class__.__name__

    def reset_stats(self) -> None:
        self.train_episode_durations = [] # list of train episode durations
        self.train_episode_indexes = [] # list of train episode indexes matching eavh evaluation phase
        self.eval_episode_durations = [] # list of lists of episode durations for each evaluation phase
    
    def get_config_dict(self) -> dict:
        return {
            "n_episodes": self.n_episodes,
            "discount_factor": self.discount_factor,
            "final_epsilon": self.final_epsilon,
        }
    
    def eval(self, n_episodes: int, env: gym.Env = None, agent: Agent = None, verbose: bool = False) -> list[int]:
        """
        Evaluate the agent with n_episodes in the environment by taking greedy actions and returns the lenghts of these eval episodes.
        """

        env = self.env if env is None else env
        agent = self.agent if agent is None else agent

        pbar = range(n_episodes)
        if verbose:
            pbar = tqdm(pbar, desc=f"Eval {self} on {self.env.spec.id}")

        episode_lengths = []
        for _ in pbar:
            
            episode_lengths.append(0)
            obs, _ = env.reset()
            terminated = False
            while not terminated:
                action = agent.policy(obs) # greedy action
                obs, _, terminated, _, _ = env.step(action)
                episode_lengths[-1] += 1
                if self.max_episode_length_eval is not None and episode_lengths[-1] >= self.max_episode_length_eval:
                    break

        return episode_lengths
    
    def save_eval_episode_durations_plot(self, path: str = None) -> None:
        """ Save plot of evaluation episode durations. """

        if len(self.train_episode_indexes) == 0 or len(self.eval_episode_durations) == 0:
            print("No evaluation episode durations to plot...")
            return
        
        path = DEFAULT_OUTPUTS_PATH if path is None else path
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # curve
        plt.plot(self.train_episode_indexes, [np.mean(lengths) for lengths in self.eval_episode_durations])

        # config parameters
        for k, v in self.get_config_dict().items():
            plt.scatter([], [], label=f"{k} = {v}", color="white")

        # axis, title and legend
        plt.xlabel("Training episode index")
        plt.ylabel(f"Averaged episode duration (limit={self.max_episode_length_eval})")
        plt.title(f"Evaluation episode duration over training")
        plt.legend(loc="lower right", handletextpad=0, handlelength=0, fontsize=8)

        # save plot
        plt.savefig(os.path.join(path, f"{self.experiment_name}_eval_durations.png"))
        plt.close()

        # save data in json
        with open(os.path.join(path, f"{self.experiment_name}_eval_durations.json"), "w") as f:
            f.write(json.dumps({
                "experiment_name": self.experiment_name,
                "max_episode_length_eval": self.max_episode_length_eval,
                "train_episode_indexes": self.train_episode_indexes,
                "eval_episode_durations": self.eval_episode_durations,
            }))
    
    def save_train_episode_durations_plot(self, path: str = None, window: float = None) -> None:
        """ Save plot of training episode durations avegared on slots of window size. """

        if len(self.train_episode_durations) == 0:
            print("No training episode durations to plot...")
            return
        
        if window is not None and window > len(self.train_episode_durations):
            print("Window size should be smaller than the number of training episodes...")
            return
        
        path = DEFAULT_OUTPUTS_PATH if path is None else path
        Path(path).mkdir(parents=True, exist_ok=True)
        
        indexes = np.arange(1, len(self.train_episode_durations) + 1)
        if window is None: # cumulative average
            avg_episode_durations = np.cumsum(self.train_episode_durations) / indexes
            title_tag = "(cumulative average)"
        else: # moving average
            incomplete_avg = np.cumsum(self.train_episode_durations[:window-1]) / np.arange(1, window)
            complete_avg = np.convolve(self.train_episode_durations, np.ones(window), 'valid') / window
            avg_episode_durations = np.hstack((incomplete_avg, complete_avg))
            assert len(avg_episode_durations) == len(self.train_episode_durations)
            title_tag = f"(average window = {window})"

        # curve
        plt.plot(indexes, avg_episode_durations)

        # config parameters
        for k, v in self.get_config_dict().items():
            plt.scatter([], [], label=f"{k} = {v}", color="white")

        # axis, title and legend
        plt.xlabel("Training episode index")
        plt.ylabel(f"Averaged episode duration")
        plt.title(f"Train episode duration over training {title_tag}")
        plt.legend(loc="lower right", handletextpad=0, handlelength=0, fontsize=8)

        # save plot
        plt.savefig(os.path.join(path, f"{self.experiment_name}_train_durations.png"))
        plt.close()

        # save data in json
        with open(os.path.join(path, f"{self.experiment_name}_train_durations.json"), "w") as f:
            f.write(json.dumps({
                "experiment_name": self.experiment_name,
                "train_episode_durations": self.train_episode_durations,
            }))


class MCTrainer(Trainer):

    DEFAULT_EXP_NAME = "mc"
    
    def __init__(
            self, 
            n_episodes: int, 
            discount_factor: float, 
            final_epsilon: float,
            learning_rate: float = None,
            n_eval: int = None, 
            max_episode_length_eval: int = None,
        ) -> None:
        super().__init__()
        self.n_episodes = n_episodes
        self.discount_factor = discount_factor
        self.final_epsilon = final_epsilon
        self.lr = learning_rate
        self.n_eval = n_eval
        self.max_episode_length_eval = max_episode_length_eval
    
    def train(self, env: gym.Env, experiment_name: str = None, save_plots: bool = False, save_agent: bool = False) -> MCAgent:
        
        # self.env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)
        self.env = env

        self.agent = MCAgent(
            action_space_size=self.env.action_space.n,
            discount_factor=self.discount_factor,
            lr=self.lr,
        )

        self.experiment_name = experiment_name if experiment_name is not None else MCTrainer.DEFAULT_EXP_NAME
        
        self.reset_stats()
        eval_every_episode = self.n_episodes // self.n_eval if self.n_eval is not None else None
        eval_every_episode = None if eval_every_episode == 0 else eval_every_episode

        log_final_eps = np.log(self.final_epsilon)
        pbar = tqdm(range(self.n_episodes), desc=f"Train {self} on {self.env.spec.id}")
        for episode_idx in pbar:

            epsilon = np.exp(log_final_eps * episode_idx / self.n_episodes)

            obs, _ = self.env.reset()
            # action = np.random.choice([0,1]) # random action to explore the environment
            action = self.agent.policy(obs, env=self.env, epsilon=epsilon)
            next_obs, reward, terminated, _, _ = self.env.step(action)

            # define lists of S_t, A_t, R_t values [see S&B section 5.3]
            states = [obs, next_obs] # S list
            actions = [action] # A list
            rewards = [None, reward] # R list
            episode_length = 1

            while not terminated:

                action = self.agent.policy(states[-1], env=self.env, epsilon=epsilon)
                next_obs, reward, terminated, _, _ = self.env.step(action)

                states.append(next_obs)
                actions.append(action)
                rewards.append(reward)
                episode_length += 1
            
            if terminated: # the last state is probably out of the state space (because the chain terminated) so we don't need it
                states.pop()

            # at this stage states and actions should have T elements and rewards T+1 elements (uncomment the following lines to check)
            assert len(actions) == len(states)
            assert len(rewards) == len(states) + 1

            # update the agent
            self.agent.update(states, actions, rewards)
            self.train_episode_durations.append(episode_length) # store the train episode duration

            if eval_every_episode is not None and episode_idx % eval_every_episode == 0:
                self.train_episode_indexes.append(episode_idx)
                self.eval_episode_durations.append(self.eval(n_episodes=100))
                pbar.set_postfix({f"avg_eval_episode_duration": np.mean(self.eval_episode_durations[-1])})
                
        if save_plots:
            self.save_train_episode_durations_plot(window=100)
            self.save_eval_episode_durations_plot()

        if save_agent:
            self.agent.save()
        
        return self.agent


class SARSALambdaTrainer(Trainer):

    DEFAULT_EXP_NAME = "sarsa-lambda"
    
    def __init__(
            self, 
            n_episodes: int, 
            learnind_rate: float,
            trace_decay: float,
            discount_factor: float, 
            final_epsilon: float, 
            n_eval: int = None, 
            max_episode_length_eval: int = None
        ) -> None:
        super().__init__()
        self.n_episodes = n_episodes
        self.lr = learnind_rate
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        self.final_epsilon = final_epsilon
        self.n_eval = n_eval
        self.max_episode_length_eval = max_episode_length_eval
    
    def get_config_dict(self) -> dict:
        return {
            "n_episodes": self.n_episodes,
            "discount_factor": self.discount_factor,
            "final_epsilon": self.final_epsilon,
            "learning_rate": self.lr,
            "trace_decay": self.trace_decay,
        }
    
    def train(self, env: gym.Env, experiment_name: str = None, save_plots: bool = False, save_agent: bool = False) -> SARSALambdaAgent:
        
        self.env = env

        self.agent = SARSALambdaAgent(
            action_space_size=self.env.action_space.n,
            discount_factor=self.discount_factor,
            lr=self.lr,
            trace_decay=self.trace_decay,
        )

        self.experiment_name = experiment_name if experiment_name is not None else SARSALambdaTrainer.DEFAULT_EXP_NAME
        
        self.reset_stats()
        eval_every_episode = self.n_episodes // self.n_eval if self.n_eval is not None else None
        eval_every_episode = None if eval_every_episode == 0 else eval_every_episode

        log_final_eps = np.log(self.final_epsilon)
        pbar = tqdm(range(self.n_episodes), desc=f"Train {self} on {self.env.spec.id}")
        for episode_idx in pbar:

            epsilon = np.exp(log_final_eps * episode_idx / self.n_episodes)

            state, _ = self.env.reset() # S1
            action = self.agent.policy(state, env=self.env, epsilon=epsilon) # A1
            terminated = False
            episode_length = 1

            # play one episode
            while not terminated:

                next_state, reward, terminated, _, _ = self.env.step(action) # R1, S2
                next_action = self.agent.policy(next_state, env=self.env, epsilon=epsilon) # A2

                # update the agent
                self.agent.update(state, action, reward, next_state, next_action, terminated)

                state = next_state
                action = next_action
                episode_length += 1

            self.train_episode_durations.append(episode_length) # store the train episode duration

            if eval_every_episode is not None and episode_idx % eval_every_episode == 0:
                self.train_episode_indexes.append(episode_idx)
                self.eval_episode_durations.append(self.eval(n_episodes=100))
                pbar.set_postfix({f"avg_eval_episode_duration": np.mean(self.eval_episode_durations[-1])})
                
        if save_plots:
            self.save_train_episode_durations_plot(window=100)
            self.save_eval_episode_durations_plot()

        if save_agent:
            self.agent.save()
                
        return self.agent


def main():

    # MCTrainer
    mc_trainer = MCTrainer(
        n_episodes=100,
        discount_factor=1.0,
        epsilon=0.1,
    )

    print(f"{mc_trainer} is ready!")

    # SARSALambdaTrainer
    sarsa_lambda_trainer = SARSALambdaTrainer(
        n_episodes=100,
        learnind_rate=0.01,
        trace_decay=0.9,
        discount_factor=1.0,
        epsilon=0.1,
    )

    print(f"{sarsa_lambda_trainer} is ready!")


if __name__ == "__main__":
    main()