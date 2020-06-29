import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


from agent_dir.agent import Agent
from environment import Environment
from mytest import test,parse,run
use_cuda = torch.cuda.is_available()


class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q


class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n

        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        #self.target_net = nn.DataParallel(self.target_net)
        #self.online_net = nn.DataParallel(self.online_net)

        if args.test_dqn:
            self.load('dqn')

        # discounted reward
        self.GAMMA = 1.0

        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network
        self.buffer_size = 10000 # max size of replay buffer
        self.memory_counter = 0
        self.m_state = np.zeros((10000,4,84,84))
        self.m_next_state = np.zeros((10000,4,84,84))
        self.m_action = np.zeros((10000,1))
        self.m_reward = np.zeros((10000,1))
        self.EPS_START = 0.9
        self.EPS_END = 0.01
        self.EPS_DECAY = 200
        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)
        self.loss_func = nn.MSELoss()
        self.steps = 0 # num. of passed steps

        # TODO: initialize your replay buffer

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.buffer_size
        self.m_state[index] = s
        self.m_next_state[index] = s_
        self.m_action[index] = a
        self.m_reward[index] = r
        self.memory_counter += 1
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass

    def make_action(self, state, test=False):
        # TODO:
        # Implement epsilon-greedy to decide whether you want to randomly select
        # an action or not.
        # HINT: You may need to use and self.steps

        #action = self.env.action_space.sample()
        if test == True:
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
        state = state.cuda()
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps / self.EPS_DECAY)
        if test== True:
            return self.online_net(state).max(1)[1].item()
        else:
            if sample > eps_threshold:
                with torch.no_grad():
                    return self.online_net(state).max(1)[1].item()
            else:
                return torch.tensor([[random.randrange(self.num_actions)]],dtype=torch.long).cuda().item()


    def update(self):
        # TODO:
        # step 1: Sample some stored experiences as training examples.
        # step 2: Compute Q(s_t, a) with your model.
        # step 3: Compute Q(s_{t+1}, a) with target model.
        # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # step 5: Compute temporal difference loss
        # HINT:
        # 1. You should not backprop to the target model in step 3 (Use torch.no_grad)
        # 2. You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it
        #    is the terminal state.
            
        # sample batch transitions
        sample_index = np.random.choice(self.buffer_size, self.batch_size)
        b_s = torch.FloatTensor(self.m_state[sample_index]).cuda()
        b_a = torch.LongTensor(self.m_action[sample_index].astype(int)).cuda()
        b_r = torch.FloatTensor(self.m_reward[sample_index]).cuda()
        b_s_ = torch.FloatTensor(self.m_next_state[sample_index]).cuda()

        # q_eval w.r.t the action in experience
        q_eval = self.online_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        return loss

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0
        best_m = 20
        best_s = 472
        qn_reward = []
        print(self.GAMMA)
        while(True):
            t_loss = 0
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)

            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)

                # TODO: store the transition in memory
                self.store_transition(state,action,reward,next_state)
                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    t_loss  = t_loss + loss

                # TODO: update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())
                
                # save the model
                """
                if self.steps % self.save_freq == 0:
                    self.save('dqn')
                """
                self.steps += 1

            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, t_loss/self.steps))
                """ 
                # play game
                if total_reward / self.display_freq >=best_m:
                    best_m = total_reward / self.display_freq
                    print(best_m)
                    self.save('dqn')
                    
                    ## play
                    args = parse()
                    tttt = run(args)
                    print(tttt)
                    if(tttt > best_s):
                        break
                """
                if episodes_done_num <=1000 and episodes_done_num%20==0:
                    qn_reward.append([episodes_done_num,(total_reward / self.display_freq)])
                total_reward = 0
                """
                if episodes_done_num >=1000:
                    print(qn_reward)
                    file = open('dqn_gamma_1.0', 'wb')
                    pickle.dump(qn_reward, file)
                    file.close()

                    break
                """
            #if self.steps 


            episodes_done_num += 1
