import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import json
import gymnasium as gym

import text_flappy_bird_gym
from utils import DEFAULT_MODELS_PATH


class Agent:

    def __init__(
        self,
        action_space_size: int,
        discount_factor: float,
    ) -> None:
        """
        Initialize a Reinforcement Learning agent with an empty dictionary of state-action values (q_values), a learning rate and an epsilon.

        ARGUMENTS:
            - action_space_size: The number of possible actions
            - discount_factor: The discount factor for computing the Q-value
            - q_values: A dictionary of state: [action values]
        """
        self.action_space_size = int(action_space_size)
        self.discount_factor = discount_factor # gamma
        self.q_values = defaultdict(lambda: [0 for _ in range(self.action_space_size)]) # default q_value for a given state is given by np.zeros(2) = [0, 0]
    
    def __str__(self) -> str:
        return self.__class__.__name__
    
    @staticmethod
    def parse_state(str_state: str) -> tuple[int, int]:
        """ "state=(x,y)" -> (x,y) """
        if not str_state.startswith("state=") or not str_state.endswith(")"):
            raise ValueError(f"Invalid state string {str_state}")
        str_x, str_y = str_state.strip("state=(").strip(")").split(",")
        return (int(str_x), int(str_y))

    @staticmethod
    def from_pretrained(agent_filename: str, path: str = None) -> "MCAgent":
        """
        Load the agent from a json file.
        """
        path = path if path is not None else DEFAULT_MODELS_PATH
        filepath = os.path.join(path, f"{agent_filename}.json")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File {filepath} not found")
        
        with open(filepath, "r") as f:
            agent_dict = json.load(f)

        agent = MCAgent(
            action_space_size=agent_dict["action_space_size"], 
            discount_factor=agent_dict["discount_factor"],
        )
        agent.q_values = defaultdict(lambda: [0 for _ in range(agent.action_space_size)])
        for k, v in agent_dict["q_values"].items():
            agent.q_values[Agent.parse_state(k)] = np.array(v)

        return agent

    def save(self, name: str = None, path: str = None) -> None:
        """
        Save the agent in a json file.
        """
        name = name if name is not None else self.__class__.__name__
        json_filename = f"{name}.json"
        path = path if path is not None else DEFAULT_MODELS_PATH
        Path(path).mkdir(parents=True, exist_ok=True)

        serialized_q_values = {
            f"state=({k[0]},{k[1]})": list(v) for k, v in self.q_values.items()
        }

        json_dict = {
            "action_space_size": self.action_space_size,
            "discount_factor": self.discount_factor,
            "q_values": serialized_q_values,
        }

        with open(os.path.join(path, json_filename), "wb") as f:
            f.write(json.dumps(json_dict).encode("utf-8"))

        print(f"Agent {self} saved in {os.path.join(path, json_filename)}")

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class MCAgent(Agent):

    def __init__(self, action_space_size: int, discount_factor: float):
        super().__init__(action_space_size=action_space_size, discount_factor=discount_factor)
        self.mean_return = defaultdict(lambda: (0,0)) # returns[x] is (0,0) by default for any x and represents (n, R_n) [see S&B section 2.4]
    
    def policy(self, state: tuple[int, int], env: gym.Env = None, epsilon: float = None) -> int:
        """
        Returns an action following an epsilon-soft policy. If env is None and epsilon is None, the agent acts greedily (inference mode).
        """
        if env is None and epsilon is None: # act greedily (inference mode)
            return int(np.argmax(self.q_values[state]))
        
        if np.random.random() < epsilon: # with probability epsilon return a random action to explore the environment
            return env.action_space.sample()
        else: # with probability (1 - epsilon) act greedily (exploit)       
            return int(np.argmax(self.q_values[state]))
    
    def update_mean_return(self, state: tuple[int, int], action: int, return_value: float) -> None:
        """ 
        Update the mean return of the state-action pair (state, action) following the incremental mean formula [see S&B section 2.4].
        """
        (n, R_n) = self.mean_return[(state, action)]
        self.mean_return[(state, action)] = (n+1, (n * R_n + return_value) / (n+1))

    def update(
        self,
        states: list[tuple[int, int]],
        actions: list[int],
        rewards: list[float],
    ) -> None:
        """
        Updates the Q-value of an action following the Monte-Carlo Exploring Starts method [see S&B section 5.3].
        """
        state_action_pairs = list(zip(states, actions)) # need to transform into a list because zip is an iterator and will be consumed by the first for loop
        T = len(states)
        G = 0

        for t in range(T-1,-1,-1): # loop over the state-action pairs in reverse order (from T-1 to 0)
            G = self.discount_factor * G + rewards[t+1]
            if not state_action_pairs[t] in state_action_pairs[:t]: # first visit of the (state, action) pair in the episode
                state_t, action_t = state_action_pairs[t]
                self.update_mean_return(state_t, action_t, return_value=G) # update the mean return for the (state_t, action_t) pair
                self.q_values[state_t][action_t] = self.mean_return[state_action_pairs[t]][1]


class SARSALambdaAgent(Agent):

    def __init__(
            self, 
            action_space_size: int, 
            discount_factor: float,
            lr: float,
            trace_decay: float,
        ):
        super().__init__(action_space_size=action_space_size, discount_factor=discount_factor)
        self.eligibility = defaultdict(lambda: [0 for _ in range(self.action_space_size)]) # eligibility[(s,a)] is 0 by default for any pair (state,action)
        self.lr = lr
        self.trace_decay = trace_decay
        self.type = "accumulate" # "replace"

    def policy(self, state: tuple[int, int], env: gym.Env = None, epsilon: float = None) -> int:
        """
        Returns an action following an epsilon-soft policy. If env is None and epsilon is None, the agent acts greedily (inference mode).
        """
        if env is None and epsilon is None: # act greedily (inference mode)
            return int(np.argmax(self.q_values[state]))
        
        if np.random.random() < epsilon: # with probability epsilon return a random action to explore the environment
            return env.action_space.sample()
        else: # with probability (1 - epsilon) act greedily (exploit)       
            return int(np.argmax(self.q_values[state]))

    def update(
        self,
        state: tuple[int, int, bool],
        action: int,
        reward: float,
        next_state: tuple[int, int, bool],
        next_action: int,
        terminated: bool,
    ) -> None:
        """
        Updates the Q-value of an action following the SARSA-lambda method [see S&B section 12.7].
        """
        q_value = self.q_values[state][action]
        next_q_value = (not terminated) * self.q_values[next_state][next_action]
        td_error = reward + self.discount_factor * next_q_value - q_value

        # sarsa-lambda
        self.eligibility[state][action] += 1
        for s in self.q_values.keys():
            for a in range(self.action_space_size):
                self.q_values[s][a] += self.lr * td_error * self.eligibility[s][a]
            if self.type == "accumulate":
                for a in range(self.action_space_size):
                    self.eligibility[s][a] *= (self.discount_factor * self.trace_decay)
            elif self.type == "replace":
                if s == state:
                    self.eligibility[s] = [0 for _ in range(self.action_space_size)]
                else:
                    for a in range(self.action_space_size):
                        self.eligibility[s][a] *= (self.discount_factor * self.trace_decay)
        # sarsa
        # self.q_values[state][action] = q_value + self.lr * td_error


def main():

    import gymnasium as gym
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)

    # MCAgent
    mc_agent = MCAgent(
        action_space_size=env.action_space.n,
        discount_factor=1.0,
    )

    print(f"{mc_agent} is ready!")

    # SARSALambdaAgent
    sarsa_lambda_agent = SARSALambdaAgent(
        action_space_size=env.action_space.n,
        discount_factor=0.95,
        lr=0.01,
        trace_decay=0.9,
    )

    print(f"{sarsa_lambda_agent} is ready!")


if __name__ == "__main__":
    main()