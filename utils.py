import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])
        
    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward = 0.0
        # gym trả về 4 giá trị, gymnasium trả về 5
        terminated, truncated = False, False 
        
        for _ in range(self._skip):
            # Cần nhận 5 giá trị từ self.env.step
            obs, reward, terminated, truncated, info = self.env.step(action)
            t_reward += reward
            # Kiểm tra cả terminated và truncated
            if terminated or truncated:
                break
        # Cần trả về 5 giá trị
        return obs, t_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None): # <<< SỬA LỖI 1: Thêm seed và options
        # Gọi reset của môi trường gốc với các tham số được truyền vào
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info # <<< SỬA LỖI 2: Trả về cả obs và info

class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(80,80,1), dtype=np.uint8)
    def observation(self, obs):
        return PreProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        # Chuyển đổi sang float để tính toán
        new_frame = frame.astype(np.float32)

        # Công thức chuẩn để chuyển sang ảnh xám
        new_frame = 0.299*new_frame[:,:,0] + 0.587*new_frame[:,:,1] + \
                    0.114*new_frame[:,:,2]

        # Cắt và giảm kích thước, sau đó thêm chiều kênh
        new_frame = new_frame[35:195:2, ::2].reshape(80,80,1)

        return new_frame.astype(np.uint8)

class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                            shape=(self.observation_space.shape[-1],
                                   self.observation_space.shape[0],
                                   self.observation_space.shape[1]),
                            dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(n_steps, axis=0),
                             env.observation_space.high.repeat(n_steps, axis=0),
                             dtype=np.float32)

    def reset(self, *, seed=None, options=None): # <<< SỬA LỖI 3: Thêm seed và options
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        # self.env.reset() bây giờ trả về (obs, info)
        obs, info = self.env.reset(seed=seed, options=options)
        # Trả về obs đã được xử lý bởi self.observation và info
        return self.observation(obs), info # <<< SỬA LỖI 4: Trả về cả obs đã xử lý và info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

def make_env(env_name, render_mode):
    env = gym.make(env_name, render_mode=render_mode)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)