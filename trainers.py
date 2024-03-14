import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
import gymnasium as gym

import text_flappy_bird_gym
from agents import Agent, MCAgent, SARSALambdaAgent
from utils import DEFAULT_OUTPUTS_PATH


class Trainer():

    def __init__(self) -> None:
        self.env = None
        self.agent = None
        self.experiment_name = None
        self.max_episode_length_eval = None
        self.episode_indexes = []
        self.episode_durations = []
    
    def __str__(self) -> str:
        return self.__class__.__name__

    def eval(self, n_episodes: int, env: gym.Env = None, agent: Agent = None, verbose: bool = False) -> None:
        """
        Evaluate the agent with n_episodes in the environment by taking greedy actions.
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
    
    def save_episode_durations_plot(self, path: str = None) -> None:
        """ Save evaluation plot of episode durations. """

        if len(self.episode_indexes) == 0 or len(self.episode_durations) == 0:
            print("No episode durations to plot...")
            return
        
        plt.plot(self.episode_indexes, [np.mean(lengths) for lengths in self.episode_durations])
        plt.xlabel("Training episode index")
        plt.ylabel(f"Average episode duration (max_limit={self.max_episode_length_eval})")
        plt.title(f"Evolution of episode duration over training with $\epsilon$={self.epsilon}")

        path = DEFAULT_OUTPUTS_PATH if path is None else path
        Path(path).mkdir(parents=True, exist_ok=True)
        # save plot
        plt.savefig(os.path.join(path, f"{self.experiment_name}_episode_durations.png"))
        # save data in json
        with open(os.path.join(path, f"{self.experiment_name}_episode_durations.json"), "w") as f:
            f.write(json.dumps({
                "episode_indexes": self.episode_indexes,
                "episode_durations": self.episode_durations,
            }))
    
    def has_training_curves(self) -> bool:
        if self.agent is not None:
            return len(self.agent.training_error) > 0
        return False
    
    def create_training_plots(self, show: bool = False, save: bool = False) -> None:

        if self.experiment_name is None:
            raise ValueError("Need to train an agent first")
    
        if len(self.agent.training_error) == 0:
            raise ValueError("Agent didn't record training error in its attribute 'training_error'")
        
        raise NotImplementedError("This method is not implemented yet")
    
    def create_value_policy_plots(self, show: bool = False, save: bool = False) -> None:

        if self.experiment_name is None:
            raise ValueError("Need to train an agent first")
        
        raise NotImplementedError("This method is not implemented yet")


class MCTrainer(Trainer):

    DEFAULT_EXP_NAME = "MC"
    
    def __init__(
            self, 
            n_episodes: int, 
            discount_factor: float, 
            epsilon: float, 
            n_eval: int = None, 
            max_episode_length_eval: int = None
        ) -> None:
        super().__init__()
        self.n_episodes = n_episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.n_eval = n_eval
        self.max_episode_length_eval = max_episode_length_eval
    
    def train(self, env: gym.Env, experiment_name: str = None, save_plots: bool = False, save_agent: bool = False) -> MCAgent:
        
        # self.env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=self.n_episodes)
        self.env = env

        self.agent = MCAgent(
            action_space_size=self.env.action_space.n,
            discount_factor=self.discount_factor,
        )

        self.experiment_name = experiment_name if experiment_name is not None else MCTrainer.DEFAULT_EXP_NAME
        
        self.episode_indexes = []
        self.episode_durations = []
        eval_every_episode = self.n_episodes // self.n_eval if self.n_eval is not None else None

        pbar = tqdm(range(self.n_episodes), desc=f"Train {self} on {self.env.spec.id}")
        for episode_idx in pbar:

            obs, _ = self.env.reset()
            random_action = np.random.choice([0,1]) # random action to explore the environment
            next_obs, reward, terminated, _, _ = self.env.step(random_action)

            # define lists of S_t, A_t, R_t values [see S&B section 5.3]
            states = [obs, next_obs] # S list
            actions = [random_action] # A list
            rewards = [None, reward] # R list

            while not terminated:

                action = self.agent.policy(states[-1], env=self.env, epsilon=self.epsilon)
                next_obs, reward, terminated, _, _ = self.env.step(action)

                states.append(next_obs)
                actions.append(action)
                rewards.append(reward)
            
            if terminated: # the last state is probably out of the state space (because the chain terminated) so we don't need it
                states.pop()

            # at this stage states and actions should have T elements and rewards T+1 elements (uncomment the following lines to check)
            assert len(actions) == len(states)
            assert len(rewards) == len(states) + 1

            # update the agent
            self.agent.update(states, actions, rewards)

            if eval_every_episode is not None and episode_idx % eval_every_episode == 0:
                self.episode_indexes.append(episode_idx)
                self.episode_durations.append(self.eval(n_episodes=100))
                pbar.set_postfix({f"avg_episode_duration": np.mean(self.episode_durations[-1])})
                
        if save_plots:
            self.save_episode_durations_plot()

        if save_agent:
            self.agent.save()
        
        return self.agent


class SARSALambdaTrainer(Trainer):

    DEFAULT_EXP_NAME = "SARSA"
    
    def __init__(
            self, 
            n_episodes: int, 
            learnind_rate: float,
            trace_decay: float,
            discount_factor: float, 
            epsilon: float, 
            n_eval: int = None, 
            max_episode_length_eval: int = None
        ) -> None:
        super().__init__()
        self.n_episodes = n_episodes
        self.lr = learnind_rate
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.n_eval = n_eval
        self.max_episode_length_eval = max_episode_length_eval
    
    def train(self, env: gym.Env, experiment_name: str = None, save_plots: bool = False, save_agent: bool = False) -> SARSALambdaAgent:
        
        self.env = env

        self.agent = SARSALambdaAgent(
            action_space_size=self.env.action_space.n,
            discount_factor=self.discount_factor,
            lr=self.lr,
            trace_decay=self.trace_decay,
        )

        self.experiment_name = experiment_name if experiment_name is not None else SARSALambdaTrainer.DEFAULT_EXP_NAME
        
        self.episode_indexes = []
        self.episode_durations = []
        eval_every_episode = self.n_episodes // self.n_eval if self.n_eval is not None else None
        eval_every_episode = None if eval_every_episode == 0 else eval_every_episode

        pbar = tqdm(range(self.n_episodes), desc=f"Train {self} on {self.env.spec.id}")
        for episode_idx in pbar:

            state, _ = self.env.reset() # S1
            action = self.agent.policy(state, env=self.env, epsilon=self.epsilon) # A1
            terminated = False

            # play one episode
            while not terminated:

                next_state, reward, terminated, _, _ = self.env.step(action) # R1, S2
                next_action = self.agent.policy(next_state, env=self.env, epsilon=self.epsilon) # A2

                self.agent.update(state, action, reward, next_state, next_action, terminated)

                state = next_state
                action = next_action

            if eval_every_episode is not None and episode_idx % eval_every_episode == 0:
                self.episode_indexes.append(episode_idx)
                self.episode_durations.append(self.eval(n_episodes=100))
                pbar.set_postfix({f"avg_episode_duration": np.mean(self.episode_durations[-1])})
                
        if save_plots:
            self.save_episode_durations_plot()

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