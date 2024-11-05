import torch 
import numpy as np 
from Deep_QN import DQN
import gymnasium as gym
import matplotlib.pyplot as plt
# from Env import CarRacing, Controller

train_environment = gym.make('CarRacing-v3', continuous=False)

max_steps = int(2e6) 
eval_interval = 10000
state_dim = (1, 128, 128) 
action_dim = 4

agent = DQN(state_dim, action_dim) 

def evaluate(n_evals=5, max_steps_per_eval=1000):  # max_steps_per_eval limits each eval episode
    scores = 0
    for i in range(n_evals):
        state, total_reward, done = train_environment.step(0), False, 0
        steps = 0  # Track steps within each episode
        while not done and steps < max_steps_per_eval:
            action = agent.act(state, training=False)
            next_state, reward, terminated = train_environment.step(action)
            state = next_state
            total_reward += reward
            done = terminated
            steps += 1  # Increment step counter
        scores += total_reward
        print(f"Evaluation {i+1}/{n_evals}, Total reward: {total_reward}, Steps: {steps}")

    return np.round(scores / n_evals, 4) 

history = {
    'Step' : [],
    'Average Return' : []
} 

state = train_environment.step(0) 

while True:
    action = agent.act(state) 
    next_state, reward, terminated = train_environment.step(action) 
    result = agent.process((state, action, reward, next_state, terminated)) 
    # print('Updated Weights...')

    state = next_state 
    if terminated:
        print('Terminated...') 
        train_environment.reset() 
        state = train_environment.step(0)

    if agent.total_steps % eval_interval == 0:
        print('Evaluation Started...') 
        score = evaluate() 
        print('Evalution Completed...')
        history['Step'].append(agent.total_steps) 
        history['Average Return'].append(score)

        plt.figure(figsize=(8, 5))
        plt.plot(history['Step'], history['Average Return'], 'r-') 
        plt.xlabel('Step', fontsize=16)
        plt.ylabel('Average Return', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis='y')
        plt.show()

        torch.save(agent.network.state_dict(), 'dqn.pt') 
        print('Saved the model...') 

    if agent.total_steps > max_steps:
        break 

