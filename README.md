# Intelligent Quadruped Robot System

An intelligent quadruped robot based on RDK (Horizon Robotics) + ROS2 Humble, integrating three core capabilities: **Reinforcement Learning Motion Control**, **Face Visual Recognition**, and **DeepSeek Voice Agent**. The robot can be controlled via gamepad for walking, and also responds to voice commands (motion control, node start/stop).

## Open Source License & Content Scope

This project is released under **Apache 2.0 License**, providing **system architecture design, core code framework, configuration and launch scripts** to help developers understand the overall pipeline design and serve as a reference for learning and further development.

### ✅ Open Source Content (Included in Repository)

- **Architecture Design Documentation**: Directory structure, service dependencies, module organization
- **Voice Agent Framework**: `deepseek_voice_agent.py` (wake word detection, VAD recording, Function Calling, node control logic)
- **Motion Control Framework**:
  - `inference_node.py` (state machine design, ROS2 subscription/publishing, stick mapping)
  - `joy_to_cmd_vel.py` (gamepad → velocity command conversion)
  - `robot_control.launch.py` (ROS2 launch configuration)
- **Visual Recognition Framework**: `bpu_insightface_recognizer.py` (ROS2 node structure, face matching pipeline)
- **Audio Playback Module**: `audio_player.py` (edge-tts integration)
- **Launch Scripts**: `start_robot.sh`, `start_bridge.sh`, `start_camera.sh`
- **WebSocket Bridge**: `ws_ros_bridge.py` (Web frontend ↔ ROS2 communication)
- **README**: Complete deployment, start/stop commands, operation instructions

### ⛔ Non-Open Source Content (Prepare Yourself)

- **RL Policy Model**: `finall.onnx` (trained weights, need to train yourself or replace)
- **Face Feature Database**: `face_embeddings.pkl` (use `generate_embeddings.py` to register new faces)
- **Vosk Speech Model**: `vosk_model/cn` (third-party open-source model, download separately)
- **ROS2 Build Artifacts**: `install/`, `build/`, `log/` (compile on target platform)
- **Hardware Low-level Driver**: `robot_bringup/` (joint communication, IMU driver, hardware-specific)
- **BPU Inference Wrapper**: `bpu_infer.py` (depends on Horizon RDK BPU toolchain)
- **Third-party Dependencies**: ONNX Runtime, RealSense driver, ROS2 packages (install per Dependencies section)

### 🎯 Quick Experience of Full Pipeline

Even without all hardware/model files, you can understand the core design from the open-source code:

| Module | What You Can Learn | Need to Prepare |
|--------|-------------------|-----------------|
| **Voice Agent** | VAD trigger logic, record-while-recognize optimization, silence early-stop, Function Calling flow, node control pattern | DeepSeek API Key, Vosk model |
| **Motion Control** | State machine design (IDLE→SQUAT→STAND→INFERENCE), button event handling, velocity mapping, 50Hz control loop | RL policy model, ROS2 environment |
| **Visual Recognition** | ROS2 node structure, face feature matching pipeline, TTS integration | RealSense camera, face feature database |
| **System Services** | systemd service configuration, node dependencies, start/stop commands | None |

**Recommended Learning Path**:
1. Read `deepseek_voice_agent.py` to understand voice wake-up + Function Calling + node control pipeline
2. Read `inference_node.py` to understand state machine design and gamepad control flow
3. Read `robot_control.launch.py` to understand ROS2 node orchestration
4. Read systemd service configurations to understand system-level service management

## Directory Structure

```
/root
├── deepseek_voice_agent.py        # Voice agent (system core, controls other nodes & robot motion)
├── camera_http_server.py          # Camera HTTP video stream service
├── ws_ros_bridge.py               # WebSocket ↔ ROS2 bridge
├── start_bridge.sh                # Start ROS bridge
├── start_camera.sh                # Start camera service
│
├── face_project/                  # ===== Visual Recognition Module =====
│   ├── bpu_nodes/
│   │   ├── bpu_insightface_recognizer.py   # Face recognition node (main entry, systemd managed)
│   │   ├── bpu_face_recognizer.py          # Face recognition (basic version)
│   │   ├── generate_embeddings.py          # Generate face feature database
│   │   ├── audio_player.py                 # TTS audio playback (edge-tts)
│   │   └── bpu_infer.py                    # BPU inference wrapper
│   ├── insightface_recognition_node.py     # InsightFace recognition node
│   ├── remote_capture.py                   # Remote stream capture
│   └── face_embeddings.pkl                 # Face feature database
│
├── legged-robot/                  # ===== Robot Motion Control Module =====
│   ├── src/
│   │   ├── robot_control/                  # Motion control package
│   │   │   ├── robot_control/
│   │   │   │   ├── inference_node.py       # State machine + ONNX policy inference (50Hz)
│   │   │   │   ├── joy_to_cmd_vel.py       # Gamepad → cmd_vel velocity command
│   │   │   │   ├── rdk_inference.py        # ONNX inference wrapper
│   │   │   │   └── finall.onnx             # Reinforcement learning policy model
│   │   │   └── launch/
│   │   │       └── robot_control.launch.py # Launch joy + joy_to_cmd_vel + inference
│   │   └── robot_bringup/                  # Low-level bridge (joint states/IMU forwarding)
│   ├── scripts/
│   │   └── start_robot.sh                  # Start bringup + control
│   └── install/                            # Build artifacts
│
└── vosk_model/cn                  # Vosk Chinese speech recognition model
```

## System Services Overview

The system is managed by 6 systemd services (auto-start on boot), each independent:

| Service Name | Description | Entry Point |
|--------------|-------------|-------------|
| `voice-agent` | Voice agent (core, controls other nodes) | `/root/deepseek_voice_agent.py` |
| `robot-inference` | Robot inference (RL policy + state machine) | `ros2 launch robot_control robot_control.launch.py` |
| `robot_control` | ROS2 low-level bridge + control | `start_robot.sh` |
| `vision-camera` | RealSense camera | `ros2 launch realsense2_camera rs_launch.py` |
| `vision-recognizer` | Face recognition node | `bpu_insightface_recognizer.py` |
| `joy_node` | Gamepad driver node | `ros2 run joy joy_node` |

## Start / Stop Commands

### Control All Services at Once

```bash
# Start all
systemctl start voice-agent robot-inference vision-camera vision-recognizer joy_node

# Stop all
systemctl stop voice-agent robot-inference vision-camera vision-recognizer joy_node

# Restart a specific service (e.g., voice)
systemctl restart voice-agent
```

### Individual Service Control

```bash
# Voice agent
systemctl start voice-agent
systemctl stop  voice-agent
journalctl -u voice-agent -f          # View real-time logs

# Robot inference (RL motion)
systemctl start robot-inference
systemctl stop  robot-inference

# Camera
systemctl start vision-camera
systemctl stop  vision-camera

# Face recognition
systemctl start vision-recognizer
systemctl stop  vision-recognizer

# Gamepad
systemctl start joy_node
systemctl stop  joy_node
```

### Boot Auto-start Management

```bash
systemctl enable  voice-agent         # Enable auto-start on boot
systemctl disable voice-agent         # Disable auto-start
systemctl status  voice-agent         # Check running status
```

## Gamepad Robot Control

The robot uses a standard ROS2 `joy` gamepad. **The `robot-inference` service must be started first** for control. The `inference_node` has a built-in state machine:

### Button Definitions

| Button | Function |
|--------|----------|
| **A button (button 0)** | Posture switch: Idle ↔ Squat ↔ Stand |
| **B button (button 1)** | Inference mode toggle: Enter/exit walking inference from standing state |

### State Flow

```
IDLE(待机) --A--> SQUAT(蹲下) --A--> STAND(站立) --A--> SQUAT
                                      |
                                      B
                                      v
                                  INFERENCE(行走推理) --B--> STAND
```

- After power-on, robot is in **IDLE**. Press **A** to squat, press **A** again to stand up.
- From standing state, press **B** to enter **INFERENCE** (walking mode). Only then the gamepad sticks become active.
- During walking, press **B** to return to standing.

### Stick Mapping (Only Active in INFERENCE Mode)

| Stick | Axis | Function | Scale |
|-------|------|----------|-------|
| Left stick vertical (up/down) | axis 1 | Forward / Backward | 0.5 m/s |
| Left stick horizontal (left/right) | axis 0 | Lateral movement | 0.5 m/s |
| Right stick horizontal (left/right) | axis 3 | In-place rotation | 1.45 rad/s |

> During walking, if velocity command is too small (< 0.15 m/s), it's automatically raised to 0.15 m/s to avoid low-speed jitter.

## Voice Agent

Location: `/root/deepseek_voice_agent.py`

### Interaction Flow

1. **Wake up**: Say **"你好小地"** (recognized as "你好小弟"), robot responds "我在，今天过得开心吗"
2. **Ask / Command**: After wake-up, speak directly. Speech is recognized, then DeepSeek is called with streaming response and TTS playback.
3. Supports **Function Calling** - voice can control robot motion and node start/stop.

### Voice Command Examples

| Phrase | Triggered Action |
|--------|------------------|
| "站起来 / 蹲下 / 休息" | Robot posture switch |
| "进入推理模式 / 准备走路" | Stand up and enter walking inference |
| "前进 / 后退 / 左转 / 右转 / 停" | Control robot motion (must first enter inference mode) |
| "启动相机 / 关闭视觉 / 启动识别" | Start/stop corresponding nodes |

### Voice-controlled Node Mapping

| Spoken Name | Corresponding Service |
|-------------|----------------------|
| 相机 (Camera) | `vision-camera.service` |
| 视觉 (Vision) / 识别 (Recognition) | `vision-recognizer.service` |
| 推理 (Inference) / 运动 (Motion) / 机器人 (Robot) | `robot-inference.service` |

### Tech Stack

- **ASR**: Vosk (Chinese offline model, `/root/vosk_model/cn`)
- **LLM**: DeepSeek (OpenAI-compatible API, streaming response)
- **TTS**: edge-tts (`zh-CN-XiaoxiaoNeural`)
- **Audio**: ES8326 codec, output volume set to max on startup

## Visual Recognition

Location: `/root/face_project/`

- **Main node**: `bpu_nodes/bpu_insightface_recognizer.py` (managed by `vision-recognizer` service)
- **Feature database**: `face_embeddings.pkl` (use `generate_embeddings.py` to add new faces)
- Depends on RealSense camera (`vision-camera` service). When a registered face is recognized, `audio_player.py` plays a voice announcement.

## Dependencies

- ROS2 Humble
- Python 3 + `openai`, `edge-tts`, `vosk`, `pyaudio`, `numpy`
- RealSense ROS2 driver (`realsense2_camera`)
- `joy` ROS2 package
- Horizon RDK BPU toolchain (visual inference)
- ONNX Runtime (RL policy inference)