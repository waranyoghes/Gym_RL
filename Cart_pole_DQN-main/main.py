import numpy as np
from dqn import DQNAGENT
if __name__ == "__main__":
    agent = DQNAgent()
    for e in range(agent.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, agent.state_size])
            done = False
            i = 0
            while not done:
                #agent.env.render()
                action = agent.act(state)
                next_state, reward, done, _ = agent.env.step(action)
                next_state = np.reshape(next_state, [1, agent.state_size])
                if not done or i == agent.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, agent.EPISODES, i, agent.epsilon))
                    if i == 500:
                        print("Saving trained model as cartpole-dqn.h5")
                        agent.save("cartpole-dqn.h5")
                        return
                agent.replay()
