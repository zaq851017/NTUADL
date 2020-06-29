import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import Environment
from agent_dir.agent import Agent
from torch.distributions import Categorical

import pickle

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        self.model = self.model.cuda()
        # self.env.observation_space.shape[0] = 8
        # self.env.action_space.n = 4
        if args.test_pg:
            self.load('pg.cpt')

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
        self.saved_log_probs = []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []
        self.saved_log_probs = []

    def make_action(self, state, test=False):
        #action = self.env.action_space.sample() # TODO: Replace this line!
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.cuda()
        probs = self.model(state)
        #print((probs))
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        # Use your model to output distribution over actions and sample from it.
        # HINT: torch.distributions.Categorical
        return action.item()

    def update(self):
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        R_i =0
        returns = []
        loss = []
        for r in self.rewards[::-1]: ## self.rewards[::-1] to reverse self.rewards
            R_i = r + self.gamma * R_i
            returns.insert(0, R_i)
        returns = torch.tensor(returns)
        #print(np.finfo(np.float32).eps.item())
        returns = (returns - returns.mean()) / (returns.std() + 0.0000001)
        #print(returns.shape)
        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))

        for log_prob, R in zip(self.saved_log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.cat(loss).sum()
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        

    def train(self):
        avg_reward = None
        best_reward = -999
        pg_1_reward = []
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                #state <numpy > < shape:8>
                #reward <float>
                #action <int>
                # 0:do nothing
                # 1:fire left orientation engine
                # 2:fire main engine
                # 3:fire right orientation engine
                self.saved_actions.append(action)
                self.rewards.append(reward)

            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1



            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
            if epoch <=2000 and epoch%40==0:
                print([epoch,avg_reward])
                pg_1_reward.append([epoch,avg_reward])
            
            if epoch >=2000:
                file = open('pg_1_reward_improvements_VR', 'wb')
                pickle.dump(pg_1_reward, file)
                file.close()
                break
            

            """
            if avg_reward > best_reward and avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                best_reward = avg_reward
                print(best_reward)
                self.save('pg.cpt')
            """
            
