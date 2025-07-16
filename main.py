import gymnasium as gym
import ale_py
from model import DeepQNetwork, Agent
from utils import plotLearning, make_env
import numpy as np
import torch

gym.register_envs(ale_py)

if __name__ == '__main__':
    env = make_env('ALE/SpaceInvaders-v5', render_mode='rgb_array') 
    brain = Agent(gamma=0.09, epsilon=1.0, alpha=0.003,
                  maxMemorySize=5000, replace=1000)
    
    while brain.memCntr < brain.memSize:
        observation, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            if done and info['lives'] == 0:
                reward = -100
            brain.storeTransition(observation, action, reward, next_observation)
            observation = next_observation
    print('Memory filled')
    
    scores = []
    epsHistory = []
    numGames = 100
    batch_size = 32
    
    for i in range(numGames):
        print('starting game', i + 1, 'epsilon: %.4f' % brain.epsilon)
        epsHistory.append(brain.epsilon)
        done = False
        observation, info = env.reset()
        score = 0
        
        while not done:
            action = brain.chooseAction(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            brain.storeTransition(observation, action, reward, next_observation)
            observation = next_observation
            brain.learn(batch_size)
            env.render()
        scores.append(score)
        print('score: ', score)
    x = [i + i for i in range(numGames)]
    filename = 'test' + str(numGames) + '.png'
    plotLearning(x, scores, epsHistory, filename=filename)
    
    # Save model
    checkpoint_file = 'best_dqn_agent.pth'
    brain.Q_eval.save_checkpoint(checkpoint_file)
    print(f"Agent saved to {checkpoint_file}")
    
    env.close()
    
    # --- INFERENCE ---
    print('\n--- Starting Evaluation ---')
    # Tạo môi trường mới với render_mode='human'
    env_eval = make_env('ALE/SpaceInvaders-v5', render_mode='human')
    
    # Tạo một Agent mới để trình diễn
    # Không cần alpha, gamma, maxMemorySize cho agent này
    # Epsilon phải là 0 (hoặc rất nhỏ) để tắt exploration
    agent_eval = Agent(gamma=0.09, epsilon=0.0, alpha=0.003, # Epsilon = 0 để tắt exploration
                       maxMemorySize=1, replace=None) # MaxMemorySize không quan trọng, replace cũng vậy
    
    # Tải trọng số đã huấn luyện vào mạng Q_eval của agent_eval
    agent_eval.Q_eval.load_checkpoint(checkpoint_file)
    # Tắt chế độ training (dropout/batchnorm, nếu có) cho mạng Q_eval
    agent_eval.Q_eval.eval() 
    
    num_eval_games = 5
    eval_scores = []

    for i in range(num_eval_games):
        print(f'Starting evaluation game {i+1}')
        done = False
        observation, info = env_eval.reset()
        score = 0

        while not done:
            action = agent_eval.chooseAction(observation)
            
            next_observation, reward, terminated, truncated, info = env_eval.step(action)
            done = terminated or truncated
            score += reward
            
            observation = next_observation
            env_eval.render() # Hiển thị game

        eval_scores.append(score)
        print(f'Evaluation score: {score}')
    
    print(f'\nAverage evaluation score over {num_eval_games} games: {np.mean(eval_scores)}')
    env_eval.close()