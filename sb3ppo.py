import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import torch
import os
import random
import numpy as np
import datagenerator as kdg
import viewportdatagenerator as vdg
import service_train_ver2
import time
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
SEED = 1
random.seed(SEED) #랜덤 시드 고정
np.random.seed(SEED)

#filesize,bitrate,qp
file_size, bitrate, q_list=kdg.generateData()

tile_popularity=kdg.generateRoiPopularity()
num_video=kdg.tot_num_video
seg_popularity=kdg.getSegsPopularityList(np.arange(num_video))
num_segs_every_video=kdg.num_segs_every_video #2s 세그먼트
#모든 비디오의 총 세그 수
tot_num_segs=sum(num_segs_every_video)
num_tile_per_seg=kdg.num_tile_per_seg
num_ver_per_tile=kdg.num_ver_per_tile
num_ver_per_seg=num_tile_per_seg*num_ver_per_tile
#모든 버전 파일 수량 계산
tot_num_vers=sum(num_segs_every_video)*num_tile_per_seg*num_ver_per_tile
vers_popularity=kdg.getTilesPopularity()
num_bw_class=20
bw_mu=11#13
bw_sigma=2#2
bandwidth_class=kdg.getTrainBandwidthClass(num_bw_class,bw_mu,bw_sigma)
#bandwidth_class=kdg.sample_logn_trunc(num_bw_class, mu_log=2.346369415862733, sigma_log=0.3114876058248091, tau=7.0, seed=1)
print(bandwidth_class)
#exit(2)
test_weight_sum=0

print('ppo')
#모든 버전 파일 수량 계산
tot_num_vers=sum(num_segs_every_video)*num_tile_per_seg*num_ver_per_tile
space_limit=kdg.space_limit

def random_select(size):
    return random.randint(0,size-1)



# ========================== 环境定义 ==========================
class MyEnv(gym.Env):
    def __init__(self):
        #super().__init__()
        super(MyEnv, self).__init__()
        #self.local_random = random.Random()
        #self.local_random.seed(time.time() + os.getpid() + id(self))
        print('training data selection')
        print('trainPopularityset generate start')
        self.trainPopularityset = kdg.generate_video_rank_train_data()
        #self.trainPopularityset=kdg.getRealDatasetRankData()
        print('trainPopularityset generate end')
        self.num_training_seg = 300 #训练样本segment 数量
        self.training_seg_list = np.random.randint(0, tot_num_segs, self.num_training_seg) #初始化时直接随机选取num_training_seg
        
        # 基本参数
        self.num_ver_per_seg = num_ver_per_seg  # 一个 segment 有多少个版本
        self.num_tile_per_seg = num_tile_per_seg  # 一个 segment 有多少 tile
        self.num_ver_per_tile = num_ver_per_tile  # 每个 tile 有多少个版本
        self.file_size = file_size  # 所有版本的文件大小
        self.q_list = q_list  # 所有版本的质量列表
        self.bitrate = bitrate  # 所有版本的码率
        self.tile_popularity = tile_popularity  # 所有 segment 的 tile 热度分布
        self.bandwidth_class = bandwidth_class  # 模拟的客户端带宽等级分布
        self.vp_tiles = vp_tiles  # 所有segment的viewport tile信息
        self.vp_bitmap = vp_bitmap  # 所有segment 的viewport bitmap，用于计算重要区域
        self.capacity = 0  # segment分配的缓存限制
        self.space_limit_rate_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #space_limit_rate  # 储存上限比例（备用）
        self.space_limit_rate_idx=0
        self.seg_no = 0  # 当前 segment 的编号
        self.init_seg_p = 0  # 当前 segment 的热度
        self.core_server_request = 0
        self.scale_factor = 100  # 奖励缩放因子
        self.select_ver_vector=np.zeros(self.num_ver_per_seg)
        # 状态维度设置：版本选择状态 + 文件大小 + q_list + 储存限制 + 已用空间 + 带宽均值方差 + tile 热度 + segment 热度
        self.state_dim = self.num_ver_per_seg * 3 + 2 + 2 + self.num_tile_per_seg + 1
        #self.state = np.zeros(self.state_dim, dtype=np.float32)
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)  # 状态空间归一化
        self.action_space = spaces.Discrete(self.num_ver_per_seg)  # 动作空间：离散选择版本
        # 状态向量中各段索引位置
        #0-num_ver_per_seg 版本选择向量
        
        self.file_size_idx = self.num_ver_per_seg
        self.qoe_idx = self.num_ver_per_seg * 2
        self.space_limit_idx = self.num_ver_per_seg * 3
        self.space_sum_idx = self.space_limit_idx + 1
        self.bw_mu_idx = self.space_sum_idx + 1
        self.bw_sigma_idx = self.bw_mu_idx + 1
        self.tiles_p_idx = self.bw_sigma_idx + 1
        self.init_seg_p_idx = self.tiles_p_idx + self.num_tile_per_seg
        self.episode_cnt=0
        self.same_seg_episode=0
        # 初始化：每个缓存率对应一个 reward 列表
        self.reward_record = [[] for _ in range(len(self.space_limit_rate_list))]
        self.space_limit_rate=None
        print('inin complete')
        if(is_train):
            self.reset()
    def setSegNo(self,_segno):
        self.seg_no=_segno
    def setSpaceLimitRate(self,_rate):
        self.space_limit_rate=_rate
    def reset(self, seed=None, options=None):
        # 初始化状态
        #tmp_seg_no = self.training_seg_list[random_select(len(self.training_seg_list))]
        #tmp_space_limit_rate_idx = random_select(len(self.space_limit_rate_list))
        #print(tmp_seg_no)
        if(is_train==True and self.same_seg_episode==0):
            #random.seed(time.time() + os.getpid() + id(self))
            self.same_seg_episode_start_time=time.time()
            self.seg_no = self.training_seg_list[random_select(len(self.training_seg_list))]
            self.space_limit_rate_idx = random_select(len(self.space_limit_rate_list))
            self.space_limit_rate=self.space_limit_rate_list[self.space_limit_rate_idx]
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.status = np.zeros(self.num_ver_per_seg, dtype=bool)  # 每个版本是否被选择

        self.select_ver_vector=np.zeros(self.num_ver_per_seg)
        self.used_capacity = 0.0  # 已用缓存容量
        self.pre_qoe = 0.0  # 上一步的视频 QoE
        self.seg_start_in_ver = self.seg_no * num_ver_per_seg
        self.seg_end_in_ver = self.seg_start_in_ver + num_ver_per_seg
        # 初始化状态字段
        self.file_size_for_episode=self.file_size[self.seg_start_in_ver:self.seg_end_in_ver]
        self.capacity=self.space_limit_rate*np.sum(self.file_size_for_episode)
        #print('capacity: ',self.capacity)
        self.q_list_for_episode=self.q_list[self.seg_start_in_ver:self.seg_end_in_ver]
        self.bitrate_for_episode=self.bitrate[self.seg_start_in_ver:self.seg_end_in_ver]
        self.file_size_max = np.max(self.file_size_for_episode)
        self.q_list_max = np.max(self.q_list_for_episode)
        self.tile_popularity_for_episode=self.tile_popularity[self.seg_no]
        self.tile_p_max=np.max(self.tile_popularity_for_episode)
        
        # 防止除以 0
        self.file_size_for_episode_norm = self.file_size_for_episode / self.file_size_max
        self.q_list_for_episode_norm = self.q_list_for_episode / self.q_list_max
        self.tile_popularity_for_episode_norm=self.tile_popularity_for_episode/self.tile_p_max
        
        
        self.state[self.file_size_idx:self.file_size_idx + self.num_ver_per_seg] = self.file_size_for_episode_norm
        self.state[self.qoe_idx:self.qoe_idx + self.num_ver_per_seg] = self.q_list_for_episode_norm
        self.state[self.space_limit_idx] = self.space_limit_rate
        self.state[self.space_sum_idx] = 0.0  # 初始使用空间为 0
        self.state[self.bw_mu_idx] = bw_mu/bw_mu # 设定固定的带宽均值和方差（可替换）
        self.state[self.bw_sigma_idx] = bw_sigma/bw_mu
        self.init_seg_p=seg_popularity[self.seg_no]
        self.state[self.init_seg_p_idx] = self.init_seg_p
        self.state[self.tiles_p_idx:self.tiles_p_idx + self.num_tile_per_seg] = self.tile_popularity_for_episode_norm
        self.vp_tiles_for_episode=vp_tiles_list[self.seg_no]
        self.vp_bitmap_for_episode=vp_bitmap[self.seg_no]
        if(is_train==True and self.same_seg_episode==0):
            self.core_server_request = service_train_ver2.service_train(np.full(num_ver_per_seg, 1), num_tile_per_seg,
                                                                   num_ver_per_tile,
                                                                   self.bitrate_for_episode,
                                                                   self.q_list_for_episode,
                                                                   self.vp_tiles_for_episode, self.vp_bitmap_for_episode,
                                                                   bandwidth_class)
        
        self.pre_gain=0

        self.p_weight=0
        self.pre_reward=0
        for period in range(24):
            # train_p_samples[period] --> sample no
            self.p_weight += self.trainPopularityset[period][self.seg_no]
        
        
        
        return self.state.copy(), {"action_mask": self.get_valid_action_mask()}

    def step(self, action):
        # 如果已选择该版本或超出缓存容量，终止本回合
        weight = self.file_size_for_episode[action]
        info={}
        if self.used_capacity + weight > self.capacity:
            done=True
            if(done):
                if(is_train==True and self.same_seg_episode%50==0):
                    print('--------------------------------')
                    print('episode',self.episode_cnt,'seg_no',self.seg_no)
                    print('same_seg_episode',self.same_seg_episode)
                    print('self.space_limit_rate',self.space_limit_rate)                
                    print('self.cache gain',self.pre_gain)
                    print('select num_action',np.sum(self.status))
                self.reward_record[self.space_limit_rate_idx].append(self.pre_gain)
                self.episode_cnt+=1
                self.same_seg_episode+=1
                if is_train==False:
                    info["select_vertor"]=self.state[:self.num_ver_per_seg].copy()
                if(is_train==True and self.same_seg_episode==300):
                    self.same_seg_episode_end_time=time.time()
                    print('seg training time',self.same_seg_episode_end_time-self.same_seg_episode_start_time)
                    self.same_seg_episode=0
                #print(self.status)
                #print(self.state[:self.num_ver_per_seg])
            return self.state.copy(), self.pre_reward, done, False, info

        # 更新状态向量与标志
        self.status[action] = True
        self.used_capacity += weight
        #print(self.used_capacity)
        self.state[action] = 1.0
        self.state[self.space_sum_idx] = self.used_capacity/self.capacity

        # 使用 service_train_ver2 中黑箱函数计算 QoE
        gain=0
        if(is_train==True):
            gain = service_train_ver2.service_train_QBver(
                self.state[:self.num_ver_per_seg],
                self.num_tile_per_seg,
                self.num_ver_per_tile,
                self.bitrate_for_episode,
                self.q_list_for_episode,
                self.vp_tiles_for_episode,
                self.vp_bitmap_for_episode,
                self.bandwidth_class,
                self.core_server_request
            )
   
            # 奖励为 QoE 增益 * segment 热度 * 缩放因子
            #reward = (gain - self.pre_gain) * self.p_weight * self.scale_factor/self.p_weight
        reward = (gain - self.pre_gain) * self.scale_factor
        self.pre_gain = gain
        self.pre_reward=reward
        done = np.sum(self.status) >= self.num_ver_per_seg  # 如果所有版本处理完，则终止
        
        if(done):
            np.set_printoptions(suppress=True, precision=6)
            if(is_train==True and self.same_seg_episode%50==0):
                print('--------------------------------')
                print('episode',self.episode_cnt,'seg_no',self.seg_no)
                print('same_seg_episode',self.same_seg_episode)
                print('self.space_limit_rate',self.space_limit_rate)
                #print('--------------------------------')
                print('episode_cnt',self.episode_cnt)
                print('self.seg_no',reward)
                print('self.cache gain',gain)
                print('select num_action',np.sum(self.status))
            self.reward_record[self.space_limit_rate_idx].append(gain)
            self.episode_cnt+=1
            self.same_seg_episode+=1
            if is_train==False:
                info["select_vertor"]=self.state[:self.num_ver_per_seg].copy()
            #print(self.status)
            if(is_train==True and self.same_seg_episode==300):
                self.same_seg_episode_end_time=time.time()
                print('seg training time',self.same_seg_episode_end_time-self.same_seg_episode_start_time)
                self.same_seg_episode=0
        return self.state.copy(), reward, done, False, info

    def get_valid_action_mask(self):
        return ~self.status  # 返回未选择的版本为合法动作
    def get_select_vertor(self):
        return self.state[:self.num_ver_per_seg]


class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # 如果一个 episode 结束，就记录该 episode 的 reward 总和
        if self.locals.get("dones") is not None and any(self.locals["dones"]):
            reward_sum = sum(self.locals["rewards"])
            self.episode_rewards.append(reward_sum)
        return True


# 在 main 函数最上方设置
#set_seed(42)
# ========================== 主程序入口 ==========================
if __name__ == '__main__':
    print('viewport data generate start')
    vp_start_time = time.time()
    vp_tiles_list=[]
    vp_bitmap=[]
    #세그먼트마다 미리 viewport생성
    for i in range(tot_num_segs):
        vp_tiles= vdg.viewportDataGenerator(num_tile_per_seg, tile_popularity[i],
                                                           len(bandwidth_class))
        bitmap=[]

        for r in range(len(bandwidth_class)):
            bitmap_per_request = []
            for j in range(num_tile_per_seg):
                if j in vp_tiles[r]:
                    bitmap_per_request.append(1)
                else:
                    bitmap_per_request.append(0)
            bitmap.append(bitmap_per_request)
        vp_bitmap.append(bitmap)
        vp_tiles_list.append(vp_tiles)
       
    print('viewport data generate end')
    print('training start')


    # ===================== 🎯 自定义 callback：记录每个 episode 的 reward =====================


    # ===================== 🛠️ 训练参数与路径设定 =====================
    is_train = False     # 是否训练
    #error_ratio = 0.1   # 热度预测误差比例
    model_path = "ppo_model_knapsack_ver9_0.6.pth"  # 模型保存路径
    env_path = "vecnormalize_knapsack_ver9_0.6.pkl" # 归一化器保存路径
    device = "cpu" if torch.backends.mps.is_available() else "cpu"

    #print("error_ratio", error_ratio)

    # ===================== 🌱 环境封装函数 =====================


    # 创建原始环
    
    def make_env():
        def _init():
            env = MyEnv()
            return ActionMasker(env, lambda env: env.get_valid_action_mask())
        return _init

    # 构造并包装环境
    num_envs = 1
    #env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    env = DummyVecEnv([make_env() for _ in range(num_envs)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=1.0)
    policy_kwargs = {
    "net_arch": [256, 256]  # 仅定义隐藏层大小
    }

    # ===================== 🤖 PPO模型构建 =====================
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        n_steps=4096,          # 一次 rollout 的步数
        batch_size=2048,       # 批处理大小
        n_epochs=20,           # 每次更新轮数
        learning_rate=3e-4,    # 学习率
        gamma=0.99,            # 折扣因子
        gae_lambda=0.95,       # GAE 参数
        clip_range=0.2,        # PPO 截断范围
        ent_coef=0.005,        # entropy 损失系数
        vf_coef=0.5,           # value function 损失系数
        max_grad_norm=0.5,     # 最大梯度裁剪
        target_kl=0.02         # KL目标
    )

    # ===================== 🚀 训练流程 =====================
    train_time_start=time.time()
    import matplotlib.pyplot as plt
    print("开始时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if is_train:
        callback = RewardLoggingCallback()
        model.learn(total_timesteps=17000000)
        train_time_end=time.time()
        print("结束时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(train_time_end-train_time_start)
        # 保存模型与归一化状态
        model.save(model_path)
        env.save(env_path)
        
        print(f"✅ 模型保存成功: {model_path}")
        print(f"✅ 环境归一化器保存成功: {env_path}")
        env_instance = env.venv.envs[0].env.unwrapped

        # 获取 reward 记录与缓存率列表
        reward_record = env_instance.reward_record
        space_limit_rate_list = env_instance.space_limit_rate_list

        # 为每个缓存率画一个图
        for i, rewards in enumerate(reward_record):
            if len(rewards) == 0:
                continue
            plt.figure()
            plt.plot(range(len(rewards)), rewards, color='orange')
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Reward Over Time (Cache Limit: {space_limit_rate_list[i]})")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
    else:
        
        # ===================== 🔁 测试流程 =====================
        print(f"📥 加载模型: {model_path}")
        model = MaskablePPO.load(model_path, device=device)

        print(f"📥 加载环境归一化器: {env_path}")
        env = DummyVecEnv([make_env() for _ in range(num_envs)])
        env = VecNormalize.load(env_path, env)
        env.training = False
        env.norm_reward = False
        rate=0.6
        tag = int(rate * 100)                 # 0.3 -> 30, 0.2 -> 20, 0.1 -> 10
        
        # 创建输出文件夹与输出文件
        os.makedirs("sb3stateinfo", exist_ok=True)
        output_path = os.path.join("sb3stateinfo", f"{int(rate*100)}state.txt")

        with open(output_path, "a") as f:
            seg_start_time=time.time()
            for seg_id in range(tot_num_segs):  # 假设有 50 个 segment
                
                env.envs[0].setSegNo(seg_id)  # 设置当前 segment 编号
                env.envs[0].setSpaceLimitRate(rate)
                
                obs = env.reset()             # 获取初始状态
                while True:
                    mask = env.env_method("get_valid_action_mask")[0]
                    a, _ = model.predict(obs, deterministic=True,action_masks=mask)
                    #print(a)
                    obs, r, dones, info = env.step(a)  # VecEnv: dones是数组
                    #print(obs)
                    if dones[0]:
                        # 优先 state[:num_ver_per_seg]，没有就用 status
                        #arr = env.envs[0].status
                        #print("info keys:", info[0].keys())
                        arr = info[0]["select_vertor"]
                        #print(arr)
                        f.write(" ".join(str(int(x)) for x in np.array(arr).reshape(-1)) + "\n")
                        break
                if((seg_id+1)%100==0):
                    seg_end_time=time.time()
                    print('seg_id',seg_id,'exec time: ',seg_end_time-seg_start_time)
                    seg_start_time=time.time()
            #exit(1)
        print(f"📄 所有版本选择已保存至 {output_path}")