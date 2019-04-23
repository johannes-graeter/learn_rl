import gym
from agents import DQNAgent
import numpy as np
from collections import deque


def main(num_episodes, render=False):
    # initialize gym environment and the agent
    env = gym.make('SpaceInvaders-v0')
    state = env.reset()
    state_shape = list(state.shape)
    state_shape[-1] = state_shape[-1] * 5
    agent = DQNAgent(state_shape, env.action_space.n)

    states = deque(maxlen=5)

    # Iterate the game
    for e in range(num_episodes):
        # reset state in the beginning of each game
        state = env.reset()
        for i in range(5):
            states.append(state)
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        num_random = 0
        train_time = 1000
        for time_t in range(train_time):
            # turn this on if you want to render
            if render:
                env.render()
            # Decide action
            action = agent.act(states)
            if agent.acted_randomly:
                num_random += 1
            # Advance the game to the next frame based on the action.
            next_state, reward, done, _ = env.step(action)
            # Remember the previous state, action, reward, and done
            agent.remember(states, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            states.append(next_state)
            # done becomes True when the game ends
            if done:
                # print the score and break out of the loop
                print(
                    "episode: {}/{}, score: {}, reward: {}, random_actions: {}%".format(e, num_episodes, time_t, reward,
                                                                                        num_random / train_time * 100))
                break
        # train the agent with the experience of the episode
        agent.replay(320)
        # print("epsilon {}".format(agent.epsilon))
        if e % 1000 == 0:
            agent.save("./deep_q_model.h5")
            print("saved model")


if __name__ == "__main__":
    main(10000)
