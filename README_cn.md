# 智能双足机器人系统

基于 RDK（地平线旭日）+ ROS2 Humble 的智能双足机器人，集成 **强化学习运动控制**、**人脸视觉识别**、**DeepSeek 语音智能体** 三大核心能力。机器人可通过手柄遥控行走，也能用语音唤醒并下达指令（控制运动、启停各功能节点）。

## 开源协议与内容范围

本项目采用 **Apache 2.0 协议**，开源内容为**系统架构设计、核心代码框架、配置与启动脚本**，旨在帮助开发者理解整体链路设计思路，可作为学习参考和二次开发基础。

### ✅ 开源内容（仓库包含）

- **架构设计文档**：目录结构、服务依赖关系、模块划分思路
- **语音智能体框架**：`deepseek_voice_agent.py`（唤醒词检测、VAD录音、Function Calling、节点控制逻辑）
- **运动控制框架**：
  - `inference_node.py`（状态机设计、ROS2订阅/发布、摇杆映射）
  - `joy_to_cmd_vel.py`（手柄→速度指令转换）
  - `robot_control.launch.py`（ROS2启动配置）
- **视觉识别框架**：`bpu_insightface_recognizer.py`（ROS2节点结构、人脸匹配流程）
- **音频播放模块**：`audio_player.py`（edge-tts集成）
- **启动脚本**：`start_robot.sh`、`start_bridge.sh`、`start_camera.sh`
- **WebSocket桥接**：`ws_ros_bridge.py`（Web前端与ROS2通信）
- **README**：完整部署、启停命令、操作说明

### ⛔ 不开源内容（需自行准备）

- **强化学习策略模型**：`finall.onnx`（训练权重文件，需自行训练或替换）
- **人脸特征数据库**：`face_embeddings.pkl`（需用 `generate_embeddings.py` 自行录入）
- **Vosk 语音模型**：`vosk_model/cn`（第三方开源模型，需自行下载）
- **ROS2 编译产物**：`install/`、`build/`、`log/`（需在目标平台编译）
- **硬件底层驱动**：`robot_bringup/`（关节通信、IMU驱动，依赖具体硬件）
- **BPU 推理封装**：`bpu_infer.py`（依赖地平线 RDK BPU 工具链）
- **第三方依赖**：ONNX Runtime、RealSense驱动、ROS2包（按依赖环境章节安装）

### 🎯 快速体验完整链路

即使不具备全部硬件/模型文件，也能从开源代码中理解核心设计思路：

| 模块 | 可学习内容 | 需自行准备 |
|------|-----------|-----------|
| **语音智能体** | VAD触发逻辑、边录边识别优化、静音早停、Function Calling调用流程、节点控制模式 | DeepSeek API Key、Vosk模型 |
| **运动控制** | 状态机设计（IDLE→SQUAT→STAND→INFERENCE）、按键事件处理、速度指令映射、50Hz控制循环 | RL策略模型、ROS2环境 |
| **视觉识别** | ROS2节点结构、人脸特征匹配流程、TTS播报集成 | RealSense相机、人脸特征库 |
| **系统服务** | systemd服务配置、节点依赖关系、启停命令 | 无 |

**推荐学习路径**：
1. 阅读 `deepseek_voice_agent.py` 理解语音唤醒+Function Calling+节点控制链路
2. 阅读 `inference_node.py` 理解状态机设计和摇杆控制流程
3. 阅读 `robot_control.launch.py` 理解 ROS2 节点编排方式
4. 阅读 systemd 服务配置理解系统级服务管理

## 目录结构

```
/root
├── deepseek_voice_agent.py        # 语音智能体（系统核心，可控制其它节点与机器人运动）
├── camera_http_server.py          # 相机 HTTP 视频流服务
├── ws_ros_bridge.py               # WebSocket ↔ ROS2 桥接
├── start_bridge.sh                # 启动 ROS 桥接
├── start_camera.sh                # 启动相机服务
│
├── face_project/                  # ===== 视觉识别模块 =====
│   ├── bpu_nodes/
│   │   ├── bpu_insightface_recognizer.py   # 人脸识别节点（主入口，systemd 拉起）
│   │   ├── bpu_face_recognizer.py          # 人脸识别（基础版）
│   │   ├── generate_embeddings.py          # 生成人脸特征库
│   │   ├── audio_player.py                 # TTS 音频播放模块（edge-tts）
│   │   └── bpu_infer.py                    # BPU 推理封装
│   ├── insightface_recognition_node.py     # InsightFace 识别节点
│   ├── remote_capture.py                   # 远程取流
│   └── face_embeddings.pkl                 # 人脸特征数据库
│
├── legged-robot/                  # ===== 机器人运动控制模块 =====
│   ├── src/
│   │   ├── robot_control/                  # 运动控制包
│   │   │   ├── robot_control/
│   │   │   │   ├── inference_node.py       # 状态机 + ONNX 策略推理（50Hz）
│   │   │   │   ├── joy_to_cmd_vel.py       # 手柄 → cmd_vel 速度指令
│   │   │   │   ├── rdk_inference.py        # ONNX 推理封装
│   │   │   │   └── finall.onnx             # 强化学习策略模型
│   │   │   └── launch/
│   │   │       └── robot_control.launch.py # 启动 joy + joy_to_cmd_vel + inference
│   │   └── robot_bringup/                  # 底层桥接（关节状态/IMU 转发）
│   ├── scripts/
│   │   └── start_robot.sh                  # 启动 bringup + control
│   └── install/                            # 编译产物
│
└── vosk_model/cn                  # Vosk 中文语音识别模型
```

## 系统服务一览

系统通过 6 个 systemd 服务管理（开机自启），互不耦合：

| 服务名 | 说明 | 入口 |
|--------|------|------|
| `voice-agent` | 语音智能体（核心，可控制其它节点） | `/root/deepseek_voice_agent.py` |
| `robot-inference` | 机器人推理（RL 策略 + 状态机） | `ros2 launch robot_control robot_control.launch.py` |
| `robot_control` | ROS2 底层桥接 + 控制 | `start_robot.sh` |
| `vision-camera` | RealSense 相机 | `ros2 launch realsense2_camera rs_launch.py` |
| `vision-recognizer` | 人脸识别节点 | `bpu_insightface_recognizer.py` |
| `joy_node` | 手柄驱动节点 | `ros2 run joy joy_node` |

## 启动 / 关闭命令

### 一键操作所有服务

```bash
# 启动全部
systemctl start voice-agent robot-inference vision-camera vision-recognizer joy_node

# 关闭全部
systemctl stop voice-agent robot-inference vision-camera vision-recognizer joy_node

# 重启某个服务（例如语音）
systemctl restart voice-agent
```

### 单个服务启停

```bash
# 语音智能体
systemctl start voice-agent
systemctl stop  voice-agent
journalctl -u voice-agent -f          # 查看实时日志

# 机器人推理（RL 运动）
systemctl start robot-inference
systemctl stop  robot-inference

# 相机
systemctl start vision-camera
systemctl stop  vision-camera

# 人脸识别
systemctl start vision-recognizer
systemctl stop  vision-recognizer

# 手柄
systemctl start joy_node
systemctl stop  joy_node
```

### 开机自启管理

```bash
systemctl enable  voice-agent         # 设置开机自启
systemctl disable voice-agent         # 取消开机自启
systemctl status  voice-agent         # 查看运行状态
```

## 遥控器操作机器人

机器人采用标准 ROS2 `joy` 手柄，**必须先启动 `robot-inference` 服务**才能控制。`inference_node` 内置状态机，按键操作如下：

### 按键定义

| 按键 | 功能 |
|------|------|
| **A 键（button 0）** | 姿态切换：待机↔蹲下↔站立 |
| **B 键（button 1）** | 推理模式开关：站立态下进入/退出行走推理 |

### 状态流转

```
IDLE(待机) --A--> SQUAT(蹲下) --A--> STAND(站立) --A--> SQUAT
                                      |
                                      B
                                      v
                                  INFERENCE(行走推理) --B--> STAND
```

- 上电后处于 **IDLE**，按 **A** 蹲下，再按 **A** 站起
- 站立状态下按 **B** 进入 **INFERENCE**（行走模式），此时摇杆才生效
- 行走中按 **B** 退回站立

### 摇杆映射（仅 INFERENCE 模式生效）

| 摇杆 | 轴 | 功能 | 缩放 |
|------|----|------|------|
| 左摇杆 纵向（上/下） | axis 1 | 前进 / 后退 | 0.5 m/s |
| 左摇杆 横向（左/右） | axis 0 | 侧移 | 0.5 m/s |
| 右摇杆 横向（左/右） | axis 3 | 原地转向 | 1.45 rad/s |

> 行走时若速度指令过小（< 0.15 m/s），会自动抬到 0.15 m/s，避免低速抖动。

## 语音智能体

文件位置：`/root/deepseek_voice_agent.py`

### 交互流程

1. **唤醒**：说 **"你好小地"**（识别为"你好小弟"），机器人应答"我在，今天过得开心吗"
2. **提问 / 下指令**：唤醒后直接说话，识别后流式调用 DeepSeek 并 TTS 播报
3. 支持 **Function Calling**，语音可控制机器人运动与节点开关

### 语音指令示例

| 说法 | 触发动作 |
|------|----------|
| "站起来 / 蹲下 / 休息" | 机器人姿态切换 |
| "进入推理模式 / 准备走路" | 站起并进入行走推理 |
| "前进 / 后退 / 左转 / 右转 / 停" | 控制机器人运动（需先进入推理模式） |
| "启动相机 / 关闭视觉 / 启动识别" | 启停对应节点 |

### 语音控制的节点映射

| 口语名称 | 对应服务 |
|----------|----------|
| 相机 | `vision-camera.service` |
| 视觉 / 识别 | `vision-recognizer.service` |
| 推理 / 运动 / 机器人 | `robot-inference.service` |

### 技术栈

- **ASR**：Vosk（中文离线模型，`/root/vosk_model/cn`）
- **LLM**：DeepSeek（OpenAI 兼容接口，流式响应）
- **TTS**：edge-tts（`zh-CN-XiaoxiaoNeural`）
- **音频**：ES8326 codec，启动时自动将输出音量拉满

## 视觉识别

文件位置：`/root/face_project/`

- **主节点**：`bpu_nodes/bpu_insightface_recognizer.py`（由 `vision-recognizer` 服务拉起）
- **特征库**：`face_embeddings.pkl`（用 `generate_embeddings.py` 录入新人脸）
- 依赖 RealSense 相机（`vision-camera` 服务），识别到已录入人脸后通过 `audio_player.py` 语音播报

## 依赖环境

- ROS2 Humble
- Python 3 + `openai`、`edge-tts`、`vosk`、`pyaudio`、`numpy`
- RealSense ROS2 驱动（`realsense2_camera`）
- `joy` ROS2 包
- 地平线 RDK BPU 工具链（视觉推理）
- ONNX Runtime（RL 策略推理）
