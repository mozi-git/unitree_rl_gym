import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        print(f"Loading policy from: {policy_path}")
        print(f"Loading xml from: {xml_path}")
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    d.qpos[7:7+num_actions] = default_angles
    
    # 打印关节信息用于调试
    print("=" * 60)
    print("关节信息调试")
    print("=" * 60)
    print(f"XML文件路径: {xml_path}")
    # print(f"模型名称: {m.model}")
    print(f"总自由度数量 (nq): {m.nq}")
    print(f"总速度数量 (nv): {m.nv}")
    print(f"控制数量 (nu): {m.nu}")
    print(f"关节数量 (njnt): {m.njnt}")
    
    # 打印所有关节名称
    print("\n所有关节名称:")
    for i in range(m.njnt):
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = m.jnt_type[i]
        joint_type_str = {0: "自由关节", 1: "球关节", 2: "滑动关节", 3: "铰链关节"}.get(joint_type, "未知")
        print(f"  关节 {i}: {joint_name} (类型: {joint_type_str})")
    
    # 打印执行器信息
    print(f"\n执行器数量: {m.nu}")
    print("执行器名称和对应的关节:")
    for i in range(m.nu):
        actuator_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        joint_id = m.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        print(f"  执行器 {i}: {actuator_name} -> 关节 {joint_id}: {joint_name}")
    
    # 打印配置中的关节顺序
    print(f"\n配置中的默认关节角度 (共{len(default_angles)}个):")
    for i, angle in enumerate(default_angles):
        print(f"  关节 {i}: {angle:.3f} rad")
    
    # 打印PD控制参数
    print(f"\nPD控制参数:")
    print(f"  Kps: {kps}")
    print(f"  Kds: {kds}")
    
    # 检查关节数量是否匹配
    if m.nq - 7 != num_actions:
        print(f"\n⚠️ 警告: 关节数量不匹配!")
        print(f"  XML模型关节数: {m.nq - 7} (总自由度{m.nq} - 7个基座自由度)")
        print(f"  配置中动作数量: {num_actions}")
    else:
        print(f"\n✅ 关节数量匹配: {m.nq - 7} = {num_actions}")
    
    print("=" * 60)

    # load policy
    policy = torch.jit.load(policy_path)
    print(f"Loading policy from: {policy_path}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # 打印当前关节位置用于调试
            if counter == 0:
                print(f"\n初始关节位置 (d.qpos[7:], 共{len(d.qpos[7:])}个):")
                for i, pos in enumerate(d.qpos[7:]):
                    print(f"  关节 {i}: {pos:.3f} rad")
            
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)