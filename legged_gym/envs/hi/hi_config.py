from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HiRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.47] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 左腿关节 (6个自由度)
            'l_hip_pitch_joint': -0.1,     # 髋部俯仰
            'l_hip_roll_joint': 0,         # 髋部滚动
            'l_hip_thigh_joint': 0,        # 髋部偏航
            'l_hip_calf_joint': 0.3,       # 膝盖
            'l_ankle_pitch_joint': -0.2,   # 踝部俯仰
            'l_ankle_roll_joint': 0,       # 踝部滚动
            
            # 右腿关节 (6个自由度)
            'r_hip_pitch_joint': -0.1,     # 髋部俯仰
            'r_hip_roll_joint': 0,         # 髋部滚动
            'r_hip_thigh_joint': 0,        # 髋部偏航
            'r_hip_calf_joint': 0.3,       # 膝盖
            'r_ankle_pitch_joint': -0.2,   # 踝部俯仰
            'r_ankle_roll_joint': 0,       # 踝部滚动
            
            # 躯干关节
            'waist_yaw_joint': 0,          # 躯干偏航
        }
    
    class env(LeggedRobotCfg.env):
        # 3(角速度) + 3(重力) + 3(命令) + 12(关节位置) + 12(关节速度) + 12(动作) + 2(相位) = 50
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

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {
            'l_hip_pitch': 25,   # 左髋部俯仰
            'l_hip_roll': 25,    # 左髋部滚动
            'l_hip_thigh': 25,   # 左髋部偏航
            'l_hip_calf': 30,    # 左膝盖
            'l_ankle_pitch': 8,  # 左踝部俯仰
            'l_ankle_roll': 8,   # 左踝部滚动
            'r_hip_pitch': 25,   # 右髋部俯仰
            'r_hip_roll': 25,    # 右髋部滚动
            'r_hip_thigh': 25,   # 右髋部偏航
            'r_hip_calf': 30,    # 右膝盖
            'r_ankle_pitch': 8,  # 右踝部俯仰
            'r_ankle_roll': 8,   # 右踝部滚动
        }
        damping = {
            'l_hip_pitch': 2.5,  # 左髋部俯仰
            'l_hip_roll': 2.5,   # 左髋部滚动
            'l_hip_thigh': 2.5,  # 左髋部偏航
            'l_hip_calf': 3.0,   # 左膝盖
            'l_ankle_pitch': 1.5, # 左踝部俯仰
            'l_ankle_roll': 1.5, # 左踝部滚动
            'r_hip_pitch': 2.5,  # 右髋部俯仰
            'r_hip_roll': 2.5,   # 右髋部滚动
            'r_hip_thigh': 2.5,  # 右髋部偏航
            'r_hip_calf': 3.0,   # 右膝盖
            'r_ankle_pitch': 1.5, # 右踝部俯仰
            'r_ankle_roll': 1.5, # 右踝部滚动
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hi/urdf/hi_12dof.urdf'
        name = "hi"
        foot_name = "ankle_roll"  # 使用踝部滚动关节作为足部
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
  ## 对于HI机器人，基础高度目标值同样设置为0.78。查看HI URDF文件：
    # 基础链接称为"base_link"，惯性原点位于xyz="0 0 -0.0394294"

    # 从base_link到脚部的腿部运动学链：
    # base_link → l_hip_pitch_link (z偏移: -0.04)
    # l_hip_pitch_link → l_hip_roll_link (z偏移: 0, y偏移: 0.07)
    # l_hip_roll_link → l_hip_thigh_link (z偏移: -0.095)
    # l_hip_thigh_link → l_hip_calf_link (z偏移: -0.097)
    # l_hip_calf_link → l_ankle_pitch_link (z偏移: -0.19945)
    # l_ankle_pitch_link → l_ankle_roll_link (z偏移: 0)
    # l_ankle_roll_link有代表脚部的碰撞几何体。

    # 从base_link到脚部接触点的总腿部长度： -0.0394294 (base_link惯性偏移) - 0.04 - 0.095 - 0.097 - 0.19945 = -0.4708794

################### g1 ########################################
#     Value function loss: 0.0007
#                     Surrogate loss: -0.0088
#              Mean action noise std: 0.80
#                        Mean reward: 0.02
#                Mean episode length: 15.08
#       Mean episode rew_action_rate: -0.0018
#             Mean episode rew_alive: 0.0020
#        Mean episode rew_ang_vel_xy: -0.0025
#       Mean episode rew_base_height: -0.0002
#           Mean episode rew_contact: 0.0023
#    Mean episode rew_contact_no_vel: -0.0010
#           Mean episode rew_dof_acc: -0.0031
#    Mean episode rew_dof_pos_limits: -0.0001
#           Mean episode rew_dof_vel: -0.0023
# Mean episode rew_feet_swing_height: -0.0003
#           Mean episode rew_hip_pos: -0.0006
#         Mean episode rew_lin_vel_z: -0.0013
#       Mean episode rew_orientation: -0.0006
#           Mean episode rew_torques: -0.0003
#  Mean episode rew_tracking_ang_vel: 0.0014
#  Mean episode rew_tracking_lin_vel: 0.0024
    
########### hi ########################################
# Value function loss: 2.5478
#                     Surrogate loss: -0.0042
#              Mean action noise std: 0.80
#                        Mean reward: -5.02
#                Mean episode length: 23.10
#       Mean episode rew_action_rate: -0.0018
#             Mean episode rew_alive: 0.0020
#        Mean episode rew_ang_vel_xy: -0.0026
#       Mean episode rew_base_height: -0.0104
#           Mean episode rew_contact: 0.0020
#    Mean episode rew_contact_no_vel: -0.0014
#           Mean episode rew_dof_acc: -0.0014
#    Mean episode rew_dof_pos_limits: -0.0005
#           Mean episode rew_dof_vel: -0.0037
# Mean episode rew_feet_swing_height: -0.0687
#           Mean episode rew_hip_pos: -0.0002
#         Mean episode rew_lin_vel_z: -0.0638
#       Mean episode rew_orientation: -0.0008
#           Mean episode rew_torques: -0.0001
#  Mean episode rew_tracking_ang_vel: 0.0018
#  Mean episode rew_tracking_lin_vel: 0.0025
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True
        soft_dof_pos_limit = 0.9
        base_height_target = 0.420  # 目标基础高度设置为0.42米
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class HiRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'hi'