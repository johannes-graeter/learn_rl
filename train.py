import gym
from agents import DQNAgent
import numpy as np


def main(num_episodes, render=False):
    # initialize gym environment and the agent
    env = gym.make('SpaceInvaders-v0')
    state = env.reset()
    agent = DQNAgent(state.shape, env.action_space.n)
    # Iterate the game
    for e in range(num_episodes):
        # reset state in the beginning of each game
        state = env.reset()
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        num_random = 0
        for time_t in range(500):
            # turn this on if you want to render
            if render:
                env.render()
            # Decide action
            action = agent.act(state)
            if agent.acted_randomly:
                num_random += 1
            # Advance the game to the next frame based on the action.
            next_state, reward, done, _ = env.step(action)
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}, random_actions: {}%".format(e, num_episodes, time_t,
                                                                              num_random / 500 * 100))
                break
        # train the agent with the experience of the episode
        agent.replay(32)
        # print("epsilon {}".format(agent.epsilon))


if __name__ == "__main__":
    main(10000)
