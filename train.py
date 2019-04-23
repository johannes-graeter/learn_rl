import gym
from agents import DQNAgent
import numpy as np
from collections import deque


def main(num_episodes, render=False):
    # initialize gym environment and the agent
    # env = gym.make('SpaceInvaders-v0')
    env = gym.make('Breakout-v0')
    state = env.reset()
    state_shape = list(state.shape)
    state_shape[-1] = state_shape[-1] * 5
    agent = DQNAgent(state_shape, env.action_space.n)

    states = deque(maxlen=5)

    max_train_time = 800

    # Iterate the game
    for e in range(num_episodes):
        # reset state in the beginning of each game
        state = env.reset()
        for i in range(5):
            states.appendleft(state)
        # time_t represents each frame of the game
        num_random = 0
        total_reward = 0.
        for time_t in range(max_train_time):
            # turn this on if you want to render
            if render:
                env.render()
            # Decide action
            action = agent.act(states)
            if agent.acted_randomly:
                num_random += 1
            # Advance the game to the next frame based on the action.
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            # Remember the previous state, action, reward, and done
            agent.remember(states.copy(), action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            states.appendleft(next_state)
            # done becomes True when the game ends
            if done:
                # print the score and break out of the loop
                rand_perc = num_random / float(time_t + 1) * 100.  # Percentage of random actions.
                print(
                    "episode: {}/{}, training_time: {}, summed_reward: {}, random_actions: {}%, eps: {}".format(e,
                                                                                                                num_episodes,
                                                                                                                time_t,
                                                                                                                total_reward,
                                                                                                                rand_perc,
                                                                                                                agent.epsilon))
                # train the agent with the experience of the episode
                agent.replay(min(100, time_t))
                break
        # print("epsilon {}".format(agent.epsilon))
        if e % 1000 == 0:
            agent.save("./deep_q_model.h5")
            print("saved model")


if __name__ == "__main__":
    main(10000, render=False)
