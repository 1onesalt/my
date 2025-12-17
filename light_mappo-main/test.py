import unittest
import numpy as np
import sys
import os

from MY_ENV.envs.target_search import target_search

class TestTargetSearchEnv(unittest.TestCase):
    
    def setUp(self):
        """每个测试用例运行前都会执行"""
        print("\n=== Setting up Environment ===")
        self.n_agents = 2
        self.env = target_search(
            n_agent=self.n_agents, 
            n_target=3, 
            max_steps=50,
            x_min=-1000, x_max=1000, 
            y_min=-1000, y_max=1000
        )
        self.obs_n = self.env.reset()

    def test_observation_space_shape(self):
        """测试1: 检查 Reset 后的观测形状是否正确"""
        print("Test 1: Checking Observation Shapes...")
        
        self.assertEqual(len(self.obs_n), self.n_agents, "观测列表长度应等于智能体数量")
        
        agent_0_obs = self.obs_n[0]
        
        # 1. 检查 PHD 热力图形状 (3, 64, 64)
        self.assertIn('phd_heatmap', agent_0_obs)
        self.assertEqual(agent_0_obs['phd_heatmap'].shape, (3, 64, 64), 
                         f"热力图形状错误, 应为 (3, 64, 64), 实测 {agent_0_obs['phd_heatmap'].shape}")
        
        # 2. 检查效用图形状 (2, 20, 20)
        self.assertIn('utility', agent_0_obs)
        self.assertEqual(agent_0_obs['utility'].shape, (2, 20, 20),
                         f"效用图形状错误, 应为 (2, 20, 20), 实测 {agent_0_obs['utility'].shape}")
        
        # 3. 检查自身位置形状 (2,)
        self.assertIn('self_pos', agent_0_obs)
        self.assertEqual(agent_0_obs['self_pos'].shape, (2,),
                         "self_pos 形状错误")

        print("-> Observation shapes passed.")

    def test_step_mechanism(self):
        """测试2: 检查 Step 函数的基本输入输出"""
        print("Test 2: Checking Step Mechanism...")
        
        # 生成随机动作
        actions = [self.env.action_space[i].sample() for i in range(self.n_agents)]
        
        next_obs, rewards, dones, infos = self.env.step(actions)
        
        # 检查返回长度
        self.assertEqual(len(next_obs), self.n_agents)
        self.assertEqual(len(rewards), self.n_agents)
        # 检查 dones 是一个列表，且长度等于智能体数量
        self.assertEqual(len(dones), self.n_agents, "Dones列表长度应等于智能体数量")
        # 或者检查所有 done 状态一致
        self.assertTrue(all(dones) == dones[0], "所有智能体的 Done 状态应一致")
        self.assertEqual(len(infos), self.n_agents)
        
        # 检查奖励类型
        self.assertIsInstance(rewards[0], float, "奖励应该是 float 类型")
        
        print("-> Step mechanism passed.")

    def test_agent_movement_and_heading(self):
        """测试3: 检查智能体移动和朝向更新逻辑"""
        print("Test 3: Checking Movement and Heading...")
        
        # 获取初始状态
        initial_pos = self.env.agent_pos[0]
        
        # 强制执行动作 0 (UP, 对应 y 增加, 朝向 90度)
        actions = [0] * self.n_agents
        self.env.step(actions)
        
        new_pos = self.env.agent_pos[0]
        new_heading = self.env.agent_headings[0]
        
        # 验证位置变化 (y 应该增加)
        self.assertGreater(new_pos[1], initial_pos[1], "执行 UP 动作后，Y 坐标应增加")
        
        # 验证朝向变化 (UP 对应 90度)
        self.assertEqual(new_heading, 90.0, f"执行 UP 动作后，朝向应为 90.0, 实测 {new_heading}")

        # 强制执行动作 3 (RIGHT, 对应 x 增加, 朝向 0度)
        actions = [3] * self.n_agents
        self.env.step(actions)
        
        final_heading = self.env.agent_headings[0]
        self.assertEqual(final_heading, 0.0, f"执行 RIGHT 动作后，朝向应为 0.0, 实测 {final_heading}")
        
        print("-> Movement and Heading logic passed.")

    def test_utility_map_logic(self):
        """测试4: 检查效用图的增长与重置逻辑"""
        print("Test 4: Checking Utility Map Growth & Reset...")
        
        # Reset 后，全图应该是 0 (因为我们在 reset 里调用了一次 update 把出生点清零了) 
        # 或者 1.0 (如果 reset 逻辑变了)。根据你最新的代码，未观测区应该是 1.0
        
        # 这里的测试策略：
        # 1. 记录 step 0 的未观测区域的效用值
        # 2. step 一次
        # 3. 检查未观测区域的值是否增加了 (growth_rate = 0.01)
        
        # 获取 agent 0 的效用图 Channel 0 (Value)
        utility_map_t0 = self.env.utility_map[0][0].copy()
        
        # 找到一个离智能体很远的点 (肯定在视域外)，假设智能体初始是随机的，我们取角落
        # 为了稳妥，我们直接看 mask
        # 重新模拟一下 mask 计算来找一个 index
        M, N = 20, 20
        pos = self.env.agent_pos[0]
        # 简单粗暴：直接 Step 一次，然后对比全图总和的变化
        # 理论上：
        #   视域内：保持 0
        #   视域外：增加 0.01
        # 所以 Sum(T1) 应该 > Sum(T0) (除非全图都在视域内，这不可能)
        
        actions = [8] * self.n_agents # NOOP
        self.env.step(actions)
        
        utility_map_t1 = self.env.utility_map[0][0].copy()
        
        # 找出 T0 时刻大于 0 的点（未观测点）
        unseen_indices = np.where(utility_map_t0 > 0.5) # 初始是 1.0
        
        if len(unseen_indices[0]) > 0:
            # 取第一个未观测点
            idx_y, idx_x = unseen_indices[0][0], unseen_indices[1][0]
            
            val_t0 = utility_map_t0[idx_y, idx_x]
            val_t1 = utility_map_t1[idx_y, idx_x]
            
            # 考虑到 clip 到 1.0，如果初始就是 1.0，增长后还是 1.0
            # 你的代码里 reset 设为 1.0。
            # 所以这里主要测试：它是否保持在 1.0，或者我们手动把某个点设小一点来测试增长
            
            # 手动注入测试：把某个角落设为 0.5
            self.env.utility_map[0][0][0, 0] = 0.5
            # 确保智能体不在 (0,0) (虽然概率极低，但逻辑严谨)
            # 假设智能体在中间，(0,0)肯定在视域外
            
            # 再次 step
            self.env.step(actions)
            val_after_step = self.env.utility_map[0][0][0, 0]
            
            # 验证增长: 0.5 -> 0.51
            self.assertAlmostEqual(val_after_step, 0.51, places=5, 
                                   msg="未观测区域应随时间增长 (+0.01)")
            
        print("-> Utility Map logic passed.")

    def test_phd_heatmap_content(self):
        """测试5: 检查 PHD 热力图是否包含有效数据"""
        print("Test 5: Checking PHD Heatmap Content...")
        
        # 这个测试比较难，因为 PHD 需要量测才能收敛。
        # 我们主要检查它不是 NaN 且形状正确。
        
        obs = self.obs_n[0]['phd_heatmap']
        
        # 检查是否有 NaN
        self.assertFalse(np.isnan(obs).any(), "PHD 热力图包含 NaN 值")
        self.assertFalse(np.isinf(obs).any(), "PHD 热力图包含 Inf 值")
        
        # 检查数值范围 (归一化后应该在 -1 到 1 之间，或者 log 后合理范围)
        # 你的代码中 ch1, ch2 做了 clip(-1, 1)
        self.assertTrue((obs[1:] >= -1.0).all() and (obs[1:] <= 1.0).all(), 
                        "PHD 速度通道 (Ch1, Ch2) 数值超出 [-1, 1] 范围")
        
        print("-> PHD Heatmap sanity check passed.")

if __name__ == '__main__':
    unittest.main()