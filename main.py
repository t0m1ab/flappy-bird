import os
import sys
import time
import gymnasium as gym
import argparse

from agents import Agent, MCAgent, SARSALambdaAgent
from trainers import MCTrainer, SARSALambdaTrainer


def demo(agent_filename: str):

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    # env = gym.make('TextFlappyBird-screen-v0', height = 15, width = 20, pipe_gap = 4)

    # initiate agent
    agent = Agent.from_pretrained(agent_filename)

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
        time.sleep(0.1) # FPS

        # If player is dead break
        if done:
            break

    env.close()


def main():

    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)

    EPISODES = 2000
    DISCOUNT_FACTOR = 1.0
    FINAL_EPSILON = 0.01
    LEARNING_RATE = 0.1
    TRACE_DECAY = 0.9

    EXPERIMENT_NAME = None

    parser = argparse.ArgumentParser(description="Train an agent to play Flappy Bird.")
    parser.add_argument(
        "--mc",
        action="store_true",
        help="use this flag to train a Monte Carlo agent",
    )
    parser.add_argument(
        "--sarsa-lambda",
        action="store_true",
        help="use this flag to train a SARSA-lambda agent",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="use this flag to launch a demo of a trained agent",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=EXPERIMENT_NAME,
        help="name of the experiment",
    )

    args = parser.parse_args()

    if args.mc and args.sarsa_lambda:
        raise ValueError("Please specify only one agent type")
    elif not args.mc and not args.sarsa_lambda:
        raise ValueError("Please specify at least one agent type")
    
    if args.demo:
        if args.mc:
            demo("MCAgent")
        elif args.sarsa_lambda:
            demo("SARSALambdaAgent")
        return
    
    if args.mc: # MC
        trainer = MCTrainer(
            n_episodes=EPISODES,
            discount_factor=DISCOUNT_FACTOR,
            final_epsilon=FINAL_EPSILON,
            learning_rate=None,
            n_eval=100,
            max_episode_length_eval=1000,
        )
    elif args.sarsa_lambda: # SARSALambda
        trainer = SARSALambdaTrainer(
            n_episodes=EPISODES,
            learnind_rate=LEARNING_RATE,
            trace_decay=TRACE_DECAY,
            discount_factor=DISCOUNT_FACTOR,
            final_epsilon=FINAL_EPSILON,
            n_eval=100,
            max_episode_length_eval=1000,
        )

    _ = trainer.train(
        env=env,
        experiment_name=args.experiment_name,
        save_plots=True,
        save_agent=True,
    )
    

if __name__ == "__main__":
    main()