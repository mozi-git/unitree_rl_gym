from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GR1T1RoughCfg( LeggedRobotCfg ):
    class init_state1( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.85] # x,y,z [m] - GR1T1站立高度约0.85米
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 左腿关节 (6个自由度)
            'left_hip_roll_joint': 0,        # 髋部滚动
            'left_hip_yaw_joint': 0,         # 髋部偏航
            'left_hip_pitch_joint': -0.3,    # 髋部俯仰
            'left_knee_pitch_joint': 0.6,    # 膝盖
            'left_ankle_pitch_joint': -0.3,  # 踝部俯仰
            'left_ankle_roll_joint': 0,      # 踝部滚动
            
            # 右腿关节 (6个自由度)
            'right_hip_roll_joint': 0,       # 髋部滚动
            'right_hip_yaw_joint': 0,        # 髋部偏航
            'right_hip_pitch_joint': -0.3,   # 髋部俯仰
            'right_knee_pitch_joint': 0.6,   # 膝盖
            'right_ankle_pitch_joint': -0.3, # 踝部俯仰
            'right_ankle_roll_joint': 0,     # 踝部滚动
        }
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.85]
        default_joint_angles = {
            # 左腿关节 - 优化为更有利于行走的初始姿态
            'left_hip_roll_joint': 0.05,      # 轻微外展，增加稳定性
            'left_hip_yaw_joint': 0.02,       # 轻微外旋，准备迈步
            'left_hip_pitch_joint': -0.6,     # 更多前倾（从-0.3增加到-0.6，约-34°）
            'left_knee_pitch_joint': 0.8,     # 更多弯曲（从0.6增加到0.8，约46°）
            'left_ankle_pitch_joint': -0.5,   # 更多背屈（从-0.3增加到-0.5，约-29°）
            'left_ankle_roll_joint': 0.02,    # 轻微内翻
            
            # 右腿关节 - 对称但略有差异，促进交替步态
            'right_hip_roll_joint': -0.05,    # 轻微外展
            'right_hip_yaw_joint': -0.02,     # 轻微外旋
            'right_hip_pitch_joint': -0.55,   # 稍少前倾，形成步态相位差
            'right_knee_pitch_joint': 0.75,   # 稍少弯曲
            'right_ankle_pitch_joint': -0.45, # 稍少背屈
            'right_ankle_roll_joint': -0.02,  # 轻微外翻
        }
    class env(LeggedRobotCfg.env):
        # 3(角速度) + 3(重力) + 3(命令) + 12(关节位置) + 12(关节速度) + 12(动作) + 2(相位) = 47
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class control1( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters for GR1T1:
        stiffness = {
            'left_hip_roll_joint': 15,     # 进一步降低刚度
            'left_hip_yaw_joint': 12,      # 左髋部偏航
            'left_hip_pitch_joint': 20,    # 左髋部俯仰
            'left_knee_pitch_joint': 25,   # 左膝盖
            'left_ankle_pitch_joint': 8,   # 左踝部俯仰
            'left_ankle_roll_joint': 6,    # 左踝部滚动
            'right_hip_roll_joint': 15,    # 右髋部滚动
            'right_hip_yaw_joint': 12,     # 右髋部偏航
            'right_hip_pitch_joint': 20,   # 右髋部俯仰
            'right_knee_pitch_joint': 25,  # 右膝盖
            'right_ankle_pitch_joint': 8,  # 右踝部俯仰
            'right_ankle_roll_joint': 6,   # 右踝部滚动
        }

        damping = {
            'left_hip_roll_joint': 0.3,    # 大幅降低阻尼系数
            'left_hip_yaw_joint': 0.25,    # 左髋部偏航
            'left_hip_pitch_joint': 0.4,   # 左髋部俯仰
            'left_knee_pitch_joint': 0.5,  # 左膝盖
            'left_ankle_pitch_joint': 0.2, # 左踝部俯仰
            'left_ankle_roll_joint': 0.15, # 左踝部滚动
            'right_hip_roll_joint': 0.3,   # 右髋部滚动
            'right_hip_yaw_joint': 0.25,   # 右髋部偏航
            'right_hip_pitch_joint': 0.4,  # 右髋部俯仰
            'right_knee_pitch_joint': 0.5, # 右膝盖
            'right_ankle_pitch_joint': 0.2,# 右踝部俯仰
            'right_ankle_roll_joint': 0.15,# 右踝部滚动
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.08  # 进一步降低动作幅度
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 降低控制频率，减少抖动

    class control( LeggedRobotCfg.control ):
        stiffness = {
            'hip_roll': 40.0, 'hip_yaw': 30.0, 'hip_pitch': 80.0,
            'knee_pitch': 120.0, # 关键：大幅提升膝盖支撑力
            'ankle_pitch': 20.0, 'ankle_roll': 20.0
        }
        damping = {
            'hip_roll': 2.0, 'hip_yaw': 1.0, 'hip_pitch': 2.0,
            'knee_pitch': 3.0, # 增加阻尼减少震荡
            'ankle_pitch': 1.0, 'ankle_roll': 1.0
        }
        # 增加动作幅度，让 225Nm 的关节真正动起来
        action_scale = 0.25

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t1/urdf/gr1t1_12dof.urdf'
        name = "gr1t1"
        foot_name = "ankle_roll"  # 使用足部滚动关节作为足部
        penalize_contacts_on = ["thigh", "shank"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True
        soft_dof_pos_limit = 0.9
        base_height_target = 0.80  # GR1T1目标基础高度设置为0.80米
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 8.0      # 大幅提升跟踪奖励权重
            tracking_ang_vel = 4.0      # 提升角速度跟踪
            lin_vel_z = -0.5            # 大幅降低z轴速度惩罚
            ang_vel_xy = -0.01          # 大幅降低xy角速度惩罚
            orientation = -0.2          # 大幅降低姿态惩罚
            base_height = -2.0          # 大幅降低基础高度惩罚
            dof_acc = -5.0e-8           # 大幅降低关节加速度惩罚
            dof_vel = -2.0e-4           # 大幅降低关节速度惩罚
            feet_air_time = 1.0         # 大幅提升足部空中时间奖励
            collision = -0.05           # 轻微碰撞惩罚
class GR1T1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01  # 增加熵系数促进探索
        learning_rate = 1e-4  # 降低学习率避免震荡
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_gr1t1_5'
        max_iterations = 2000  # 延长训练时间
        save_interval = 100