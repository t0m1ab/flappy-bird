import os
import sys
import time
import gymnasium as gym

from agents import Agent, MCAgent, SARSALambdaAgent
from trainers import Trainer, MCTrainer, SARSALambdaTrainer


def demo(agent_filename: str):

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 12)
    # env = gym.make('TextFlappyBird-screen-v0', height = 15, width = 20, pipe_gap = 4)

    # initiate agent
    agent = MCAgent.from_pretrained(agent_filename)

    # episode
    obs, info = env.reset()
    while True:

        # Select next action
        # action = env.action_space.sample() # for an agent, action = agent.policy(observation)
        action = agent.policy(obs)

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)

        # Render the game
        os.system("clear")
        sys.stdout.write(env.render())
        time.sleep(0.2) # FPS

        # If player is dead break
        if done:
            break

    env.close()


def main():

    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)

    EPISODES = 5000
    DISCOUNT_FACTOR = 1.0
    EPSILON = 0.1
    LEARNING_RATE = 0.1
    TRACE_DECAY = 0.9

    EXPERIMENT_NAME = None

    # MC
    trainer = MCTrainer(
        n_episodes=EPISODES,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        n_eval=100,
        max_episode_length_eval=1000,
    )

    # SARSALambda
    # trainer = SARSALambdaTrainer(
    #     n_episodes=EPISODES,
    #     learnind_rate=LEARNING_RATE,
    #     trace_decay=TRACE_DECAY,
    #     discount_factor=DISCOUNT_FACTOR,
    #     epsilon=EPSILON,
    #     n_eval=100,
    #     max_episode_length_eval=1000,
    # )

    agent = trainer.train(
        env=env,
        experiment_name=EXPERIMENT_NAME,
        save_plots=True,
        save_agent=True,
    )
    

if __name__ == "__main__":
    # main()
    demo("MCAgent")
    # demo("SARSALambdaAgent")