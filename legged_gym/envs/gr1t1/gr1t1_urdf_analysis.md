# GR1T1机器人URDF文件分析

## 文件基本信息
- **文件名**: gr1t1_12dof.urdf
- **机器人名称**: g1r1_12dof
- **自由度**: 12个（左右腿各6个）
- **文件路径**: resources/robots/gr1t1/urdf/gr1t1_12dof.urdf

## 机器人整体结构

### 基座 (Base Link)
- **链接名称**: base_link
- **质量**: 3.84 kg
- **惯性中心**: xyz="-0.06584 -9.765E-5 -0.01434"
- **碰撞几何**: 长方体 (0.15m × 0.25m × 0.1m)
- **视觉几何**: base_link.STL 模型

### 浮动基座关节
- **关节名称**: floating_base_joint
- **类型**: 浮动关节 (6自由度)
- **父链接**: world
- **子链接**: base_link

## 左腿关节链结构

### 1. 左髋部滚动关节 (Left Hip Roll)
- **关节名称**: left_hip_roll_joint
- **类型**: 旋转关节 (revolute)
- **轴方向**: x轴 (1, 0, 0)
- **运动范围**: -0.09 rad 到 0.79 rad
- **最大力矩**: 48 N·m
- **最大速度**: 12.15 rad/s
- **对应连杆**: left_thigh_roll_link
- **质量**: 1.45 kg

### 2. 左髋部偏航关节 (Left Hip Yaw)
- **关节名称**: left_hip_yaw_joint
- **类型**: 旋转关节 (revolute)
- **轴方向**: z轴 (0, 0, 1)
- **运动范围**: -0.7 rad 到 0.7 rad
- **最大力矩**: 66 N·m
- **最大速度**: 16.76 rad/s
- **对应连杆**: left_thigh_yaw_link
- **质量**: 3.17 kg

### 3. 左髋部俯仰关节 (Left Hip Pitch)
- **关节名称**: left_hip_pitch_joint
- **类型**: 旋转关节 (revolute)
- **轴方向**: y轴 (0, 1, 0)
- **运动范围**: -1.75 rad 到 0.7 rad
- **最大力矩**: 225 N·m
- **最大速度**: 37.38 rad/s
- **对应连杆**: left_thigh_pitch_link
- **质量**: 7.99 kg

### 4. 左膝盖关节 (Left Knee)
- **关节名称**: left_knee_pitch_joint
- **类型**: 旋转关节 (revolute)
- **轴方向**: y轴 (0, 1, 0)
- **运动范围**: -0.09 rad 到 1.92 rad
- **最大力矩**: 225 N·m
- **最大速度**: 37.38 rad/s
- **对应连杆**: left_shank_pitch_link
- **质量**: 1.93 kg

### 5. 左踝部俯仰关节 (Left Ankle Pitch)
- **关节名称**: left_ankle_pitch_joint
- **类型**: 旋转关节 (revolute)
- **轴方向**: y轴 (0, 1, 0)
- **运动范围**: -1.05 rad 到 0.52 rad
- **最大力矩**: 15 N·m
- **最大速度**: 20.32 rad/s
- **对应连杆**: left_foot_pitch_link
- **质量**: 0.538 kg

### 6. 左踝部滚动关节 (Left Ankle Roll)
- **关节名称**: left_ankle_roll_joint
- **类型**: 旋转关节 (revolute)
- **轴方向**: x轴 (1, 0, 0)
- **运动范围**: -0.44 rad 到 0.44 rad
- **最大力矩**: 30 N·m
- **最大速度**: 20.32 rad/s
- **对应连杆**: left_foot_roll_link
- **质量**: 0.538 kg
- **碰撞几何**: 两个圆柱体 (长度0.24m, 半径0.02m)

## 右腿关节链结构（与左腿对称）

### 1. 右髋部滚动关节 (Right Hip Roll)
- **关节名称**: right_hip_roll_joint
- **运动范围**: -0.79 rad 到 0.09 rad
- **对应连杆**: right_thigh_roll_link

### 2. 右髋部偏航关节 (Right Hip Yaw)
- **关节名称**: right_hip_yaw_joint
- **对应连杆**: right_thigh_yaw_link

### 3. 右髋部俯仰关节 (Right Hip Pitch)
- **关节名称**: right_hip_pitch_joint
- **对应连杆**: right_thigh_pitch_link

### 4. 右膝盖关节 (Right Knee)
- **关节名称**: right_knee_pitch_joint
- **对应连杆**: right_shank_pitch_link

### 5. 右踝部俯仰关节 (Right Ankle Pitch)
- **关节名称**: right_ankle_pitch_joint
- **对应连杆**: right_foot_pitch_link

### 6. 右踝部滚动关节 (Right Ankle Roll)
- **关节名称**: right_ankle_roll_joint
- **对应连杆**: right_foot_roll_link

## 关键几何参数

### 基座位置
- **初始位置**: [0.0, 0.0, 0.85] 米（在配置文件中设置）

### 腿部运动学链长度
- **髋部到膝盖**: 约 0.36 米
- **膝盖到踝部**: 约 0.34 米
- **踝部到足尖**: 约 0.03 米
- **总站立高度**: 约 0.85 米

## 碰撞几何优化

### 简化策略
1. **基座**: 使用长方体替代复杂STL模型
2. **大腿/小腿**: 使用长方体进行简化
3. **足部**: 使用两个圆柱体模拟足底接触

### 碰撞几何参数
- **基座碰撞盒**: 0.15m × 0.25m × 0.1m
- **大腿碰撞盒**: 0.08m × 0.08m × 0.15m
- **小腿碰撞盒**: 0.08m × 0.08m × 0.25m
- **足部圆柱**: 长度0.24m, 半径0.02m

## 环境配置适配

### 观测空间
- **观测值数量**: 47个
  - 3个角速度
  - 3个重力向量
  - 3个命令
  - 12个关节位置
  - 12个关节速度
  - 12个动作
  - 2个相位（sin/cos）

### 动作空间
- **动作数量**: 12个（对应12个关节）

### PD控制参数
- **刚度 (Stiffness)**: 15-40 N·m/rad
- **阻尼 (Damping)**: 3.0-7.0 N·m·s/rad

## 主要特点

1. **对称设计**: 左右腿完全对称，便于控制算法设计
2. **简化碰撞**: 使用基本几何体替代复杂STL模型，提高仿真效率
3. **合理参数**: 关节力矩和速度限制基于实际机器人能力
4. **标准命名**: 关节和链接命名遵循URDF标准规范
5. **兼容性**: 设计兼容Isaac Gym仿真环境

## 使用注意事项

1. **足部识别**: 在配置文件中设置 `foot_name = "foot_roll_link"`
2. **初始姿态**: 确保关节初始角度与URDF定义一致
3. **碰撞检测**: 简化几何体可能影响精细的接触检测
4. **质量分布**: 惯性参数基于实际机器人测量数据

此URDF文件为GR1T1机器人在Isaac Gym环境中的仿真提供了完整的机器人模型定义。