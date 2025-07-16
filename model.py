import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, alpha):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128*7*7, 512)
        self.fc2 = nn.Linear(512, 6)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, observation):
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 128*7*7)
        observation = F.relu(self.fc1(observation))
        
        actions = self.fc2(observation)
        
        return actions
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
        
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
    
class Agent(object):
    def __init__(self, gamma, epsilon, alpha,
                 maxMemorySize, epsEnd=0.05,
                 replace=10000, actionSpace=[0, 1, 2, 3, 4, 5]):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsEnd = epsEnd
        self.alpha = alpha
        self.actionSpace = actionSpace
        self.memory = []
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memCntr = 0 # total memory stored
        self.replace_target_cnt = replace # how often to replace the target network
        self.Q_eval = DeepQNetwork(self.alpha)
        self.Q_next = DeepQNetwork(self.alpha)
        
    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr % self.memSize] = [state, action, reward, state_]
        self.memCntr += 1
            
    def chooseAction(self, observation):
        rand = np.random.random()
        observation_tensor = torch.Tensor(observation).to(self.Q_eval.device)
        observation_tensor = observation_tensor.unsqueeze(0)
        actions = self.Q_eval.forward(observation_tensor)
        
        if rand < 1 - self.epsilon:
            action = torch.argmax(actions, dim=1).item()
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1
        return action
    
    def learn(self, batch_size):
        # 1. Zeroing Gradients
        self.Q_eval.optimizer.zero_grad()
        
        # 2. Update Target Network
        if self.replace_target_cnt is not None and \
            self.learn_step_counter % self.replace_target_cnt == 0: # time to replace the target network
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
        
        # 3. Sample Mini-Batch from Experience Replay    
        # Sampling indices for the mini-batch
        # Fix for memCntr - batch_size - 1 might be negative if memCntr is too small
        # Better to ensure memCntr >= batch_size for sampling
        if len(self.memory) < batch_size:
            # Not enough memory to sample a full batch, maybe skip learning or wait
            return 
        
        # Select random indices for the mini-batch
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        # replace=False để tránh chọn cùng một transition nhiều lần trong cùng một batch

        # Extract components from memory using these indices
        # Khởi tạo các danh sách để thu thập dữ liệu
        states = []
        actions = []
        rewards = []
        next_states = []

        # Lặp qua các chỉ số được chọn và lấy dữ liệu
        for idx in batch_indices:
            s, a, r, s_ = self.memory[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)

        # Convert to numpy arrays, then to PyTorch Tensors
        # Quan trọng: Đảm bảo dtype là float32 cho hình ảnh và long cho hành động
        states_tensor = torch.Tensor(np.array(states)).to(self.Q_eval.device)
        # Reshape for CNN input: (batch_size, num_channels, height, width)
        # Nếu CNN của bạn có 1 kênh đầu vào: (batch_size, 1, H, W)
        # states_tensor = states_tensor.unsqueeze(1) # Add channel dimension
        
        actions_tensor = torch.Tensor(np.array(actions)).long().to(self.Q_eval.device) # Action should be Long tensor
        rewards_tensor = torch.Tensor(np.array(rewards)).to(self.Q_eval.device)
        
        next_states_tensor = torch.Tensor(np.array(next_states)).to(self.Q_eval.device)
        # Reshape for CNN input: (batch_size, num_channels, height, width)
        # next_states_tensor = next_states_tensor.unsqueeze(1) # Add channel dimension
        
        # 4. Forward Pass through Networks
        Qpred = self.Q_eval.forward(states_tensor)
        Qnext = self.Q_next.forward(next_states_tensor)
        
        # 5. Compute TD Target (Q-target)
        # Trích xuất các hành động đã thực hiện (A_t) từ mini-batch

        # Lấy Q-value của hành động ĐÃ THỰC HIỆN từ mạng Q_eval
        # Đây là giá trị Q(S_t, A_t) mà chúng ta muốn cập nhật
        Q_eval_actions = Qpred[np.arange(batch_size), actions_tensor] 

        # Tính giá trị max Q của trạng thái kế tiếp từ mạng mục tiêu, và detach nó
        max_Q_next = torch.max(Qnext, dim=1)[0].detach() 

        # Tính Q-target
        Q_target_values = rewards_tensor + self.gamma * max_Q_next

        # 6. Epsilon Decay
        if self.steps > 500:
            if self.epsilon - 1e-4 > self.epsEnd:
                self.epsilon -= 1e-4
            else:
                self.epsilon = self.epsEnd
                
        # 7. Compute Loss, Backpropagate, and Optimize
        loss = self.Q_eval.loss(Q_eval_actions, Q_target_values)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1