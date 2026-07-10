\# InferenceNode 节点完整详解



> 源文件：`inference\_node.py`

> 机器人：10 自由度腿式机器人（5 关节 × 2 腿）

> 控制频率：50Hz（dt = 0.02s）



\---



\## 目录



1\. \[节点定位与整体架构](#1-节点定位与整体架构)

2\. \[导入与依赖（L1-9）](#2-导入与依赖l1-9)

3\. \[节点初始化 \\\_\\\_init\\\_\\\_（L11-78）](#3-节点初始化-\_\_init\_\_l11-78)

4\. \[joint\_callback — 关节状态回调](#4-joint\_callback--关节状态回调)

5\. \[imu\_callback — IMU 回调（含四元数转欧拉角推导）](#5-imu\_callback--imu-回调含四元数转欧拉角推导)

6\. \[cmd\_callback — 速度指令回调](#6-cmd\_callback--速度指令回调)

7\. \[joy\_callback — 手柄回调（状态机驱动）](#7-joy\_callback--手柄回调状态机驱动)

8\. \[start\_transition — 状态过渡（线性插值）](#8-start\_transition--状态过渡线性插值)

9\. \[timer\_callback — 主循环回调](#9-timer\_callback--主循环回调)

10\. \[main() 入口函数（L209-217）](#10-main-入口函数l209-217)

11\. \[附录：数学公式汇总](#11-附录数学公式汇总)



\---



\## 1. 节点定位与整体架构



\### 1.1 节点定位



`InferenceNode` 继承自 `rclpy.node.Node`，节点名是 `inference\_node`。它是一个\*\*纯推理节点\*\*——本身不训练，只加载已训练好的 ONNX 模型（`finall.onnx`），根据传感器状态实时输出电机目标位置。



\### 1.2 整体数据流



```

&#x20;                         ┌─────────────────────────────────────┐

&#x20;                         │           InferenceNode            │

&#x20;                         │                                     │

&#x20; 关节编码器 ──> joint\_states ──> joint\_callback ──┐            │

&#x20; IMU       ──> imu/data     ──> imu\_callback   ──┤            │

&#x20; 遥控/语音 ──> cmd\_vel      ──> cmd\_callback   ──┼─> 状态缓存 │

&#x20; 手柄      ──> joy          ──> joy\_callback   ──┘            │

&#x20;                                                 │            │

&#x20;                         50Hz timer ──────────────┼─> 状态机  │

&#x20;                                                 │   分发    │

&#x20;                                                 ▼            │

&#x20;                                         ┌───────────────┐  │

&#x20;                                         │ RL 策略 π\_θ   │  │

&#x20;                                         │ (ONNX 推理)   │  │

&#x20;                                         └───────┬───────┘  │

&#x20;                                                 │            │

&#x20;                         motor\_cmds <────────────┘            │

&#x20;                         └─────────────────────────────────────┘

&#x20;                                 │

&#x20;                                 ▼

&#x20;                         底层 PD 控制器 ──> 电机力矩 ──> 机器人

```



\### 1.3 控制分层架构



```

\[观测 o\_t] ──> \[RL策略 π\_θ] ──> \[目标位置 a\_t] ──> \[底层PD] ──> \[电机力矩 τ] ──> \[机器人]

```



RL 在\*\*位置层面\*\*做规划，力矩由底层 PD 闭环：



$$

\\tau\_i = K\_p (a\_i - q\_i) - K\_d \\dot{q}\_i

$$



其中 $K\_p, K\_d$ 是位置增益和阻尼增益（在底层驱动中实现，本节点不可见）。



\---



\## 2. 导入与依赖（L1-9）



```python

import rclpy

from rclpy.node import Node

import numpy as np

from sensor\_msgs.msg import JointState, Imu, Joy

from std\_msgs.msg import Float64MultiArray

from geometry\_msgs.msg import Twist, Vector3

import os



from .rdk\_inference import TinkerRealInference

```



\### 依赖说明



| 导入 | 来源 | 作用 |

|------|------|------|

| `rclpy`, `Node` | ROS2 Python 客户端库 | 节点基类与生命周期管理 |

| `numpy as np` | NumPy | 矩阵/向量运算，四元数转欧拉角、插值 |

| `JointState` | sensor\_msgs | 关节位置/速度消息 |

| `Imu` | sensor\_msgs | IMU 四元数姿态 + 角速度 |

| `Joy` | sensor\_msgs | 手柄按键/摇杆 |

| `Float64MultiArray` | std\_msgs | 发布 10 维电机目标位置 |

| `Twist` | geometry\_msgs | 平面速度指令（线速度 + 角速度） |

| `TinkerRealInference` | 本地 `rdk\_inference.py` | 封装 ONNX 模型加载与推理 |



> \*\*注意\*\*：`import os` 和 `Vector3` 在当前代码中未实际使用，属于遗留导入。



\---



\## 3. 节点初始化 \_\_init\_\_（L11-78）



这是理解整个节点的核心，初始化分为以下几个部分。



\### 3.1 节点创建（L13）



```python

super().\_\_init\_\_('inference\_node')

```



调用父类 `Node` 构造函数，注册节点名 `inference\_node`，使该节点可被 ROS2 图（graph）发现。



\### 3.2 模型加载（L15-27）



```python

self.declare\_parameter('model\_path', '/root/legged-robot/src/robot\_control/robot\_control/finall.onnx')

model\_path = self.get\_parameter('model\_path').get\_parameter\_value().string\_value



self.get\_logger().info(f'Loading model from: {model\_path}')



try:

&#x20;   self.inference = TinkerRealInference(model\_path)

&#x20;   self.get\_logger().info('Model loaded successfully')

except Exception as e:

&#x20;   self.get\_logger().error(f'Failed to load model: {e}')

&#x20;   self.inference = None

```



\- 模型路径通过 ROS2 参数 `model\_path` 配置，默认指向 `finall.onnx`，可在 launch 文件中覆盖。

\- 实际推理逻辑封装在 `rdk\_inference.py` 的 `TinkerRealInference` 类中。

\- 加载失败不会直接退出，而是把 `self.inference` 置为 `None`，后续 `timer\_callback` 第一行 `if self.inference is None: return` 会静默跳过——这是一种容错，但也会让节点"假活"，建议失败时直接 `sys.exit`。



\### 3.3 状态观测变量（L29-38）



这是强化学习策略 $\\pi$ 的观测向量原料：



| 变量 | 维度 | 含义 | 初始值 |

|------|------|------|--------|

| `latest\_euler` | 3 | IMU 欧拉角 \[roll, pitch, yaw] | `np.zeros(3)` |

| `latest\_imu\_gyro` | 3 | IMU 角速度 \[ωx, ωy, ωz] | `\[0,0,0]` |

| `latest\_joint\_pos` | 10 | 关节当前角度 | `np.zeros(10)` |

| `latest\_joint\_vel` | 10 | 关节当前速度 | `np.zeros(10)` |

| `cmd\_vx` | 1 | 期望前进速度 | 0.0 |

| `cmd\_vy` | 1 | 期望侧移速度 | 0.0 |

| `cmd\_dyaw` | 1 | 期望偏航角速度 | 0.0 |



所有变量都用 `np.float32` 初始化，与 ONNX 模型输入类型对齐。初始值为零向量，意味着在传感器数据到达前，观测全为零。



\### 3.4 状态机定义（L40-48）



```python

self.STATE\_IDLE = -1

self.STATE\_SQUAT = 0

self.STATE\_TRANSITIONING = 1

self.STATE\_STAND = 2

self.STATE\_INFERENCE = 3



self.state = self.STATE\_IDLE

self.next\_state = self.STATE\_IDLE

```



节点用\*\*有限状态机\*\*管理安全过渡，不是一上来就跑模型：



| 状态 | 值 | 行为 |

|------|----|------|

| IDLE | -1 | 上电默认，不发任何指令，机器人保持下电姿态 |

| SQUAT | 0 | 蹲下，持续发送 `target\_pos\_squat` |

| TRANSITIONING | 1 | 过渡态，线性插值滑到目标角度 |

| STAND | 2 | 站立，持续发送 `target\_pos\_stand` |

| INFERENCE | 3 | 真正跑 RL 模型，输出动作 |



完整流转图：



```

&#x20;                   A键(2s)                A键(2s)

&#x20;   ┌──────────────────────┐    ┌──────────────────────┐

&#x20;   │                      ▼    │                      ▼

&#x20; IDLE(-1)              SQUAT(0)              STAND(2)

&#x20;   │                      ▲                      │

&#x20;   │                      │ A键(2s)               │ B键(立即)

&#x20;   │                      │ (从STAND/INFERENCE)   ▼

&#x20;   │                      │                  INFERENCE(3)

&#x20;   │                      │                      │

&#x20;   │                      └────── B键(1s) ───────┘

&#x20;   │

&#x20;   └── 上电默认状态

```



\### 3.5 目标关节位置（L50-55）



\*\*站立姿态\*\*（L50）：



```python

self.target\_pos\_stand = np.array(\[0.0, 0.08, 0.56, -1.12, -0.57, 0.0, -0.08, -0.56, 1.12, 0.57], dtype=np.float32)

```



\- 前 5 个是左腿，后 5 个是右腿。

\- 注意左右腿对应关节符号基本相反（镜像对称），如 `q1=0.08` vs `q6=-0.08`、`q2=0.56` vs `q7=-0.56`。



\*\*蹲下姿态\*\*（L52-55）：



```python

self.target\_pos\_squat = np.array(\[

&#x20;   -0.039101600646972656, 0.032998085021972656, 1.9686040878295898, -2.4729156494140625, -0.49267578125,

&#x20;   -0.00934600830078125, 0.10738563537597656, -1.8488216400146484, 2.585068702697754, 0.6300067901611328

], dtype=np.float32)

```



\- 来自用户实测数据，数值较大（如 `1.969/-2.473`），说明关节弯曲明显，是典型的蹲姿。

\- 单位为\*\*弧度（rad）\*\*。



\### 3.6 过渡变量（L57-60）



```python

self.transition\_start\_pos = None

self.transition\_target\_pos = None

self.transition\_progress = 0.0

self.transition\_step = 0.0

```



用于 `start\_transition` 的线性插值：

\- `transition\_start\_pos`：过渡起点（当前关节位置拷贝）。

\- `transition\_target\_pos`：过渡终点（目标位置拷贝）。

\- `transition\_progress`：插值进度 $p \\in \[0, 1]$。

\- `transition\_step`：每帧进度增量 $\\Delta t / T$。



\### 3.7 手柄边沿检测状态（L62-63）



```python

self.a\_pressed\_last = False

self.b\_pressed\_last = False

```



保存上一帧按键状态，用于上升沿检测（见第 7 节）。



\### 3.8 话题订阅（L65-69）



```python

self.create\_subscription(JointState, 'joint\_states', self.joint\_callback, 10)

self.create\_subscription(Imu, 'imu/data', self.imu\_callback, 10)

self.create\_subscription(Twist, 'cmd\_vel', self.cmd\_callback, 10)

self.create\_subscription(Joy, 'joy', self.joy\_callback, 10)

```



| 话题 | 消息类型 | 回调 | QoS | 数据来源 |

|------|----------|------|-----|----------|

| `joint\_states` | JointState | `joint\_callback` | 10 | 关节编码器/驱动器 |

| `imu/data` | Imu | `imu\_callback` | 10 | IMU 传感器 |

| `cmd\_vel` | Twist | `cmd\_callback` | 10 | 遥控器/键盘/语音助手 |

| `joy` | Joy | `joy\_callback` | 10 | 手柄 |



QoS（Quality of Service）深度为 10，表示缓冲 10 条消息，是可靠传输的默认值。



\### 3.9 话题发布（L72）



```python

self.motor\_cmd\_pub = self.create\_publisher(Float64MultiArray, 'motor\_cmds', 10)

```



\- 发布话题：`motor\_cmds`

\- 消息类型：`Float64MultiArray`，内含 10 维浮点数组。

\- 这是节点唯一的输出，直接驱动底层电机控制器。



\### 3.10 定时器（L74-76）



```python

self.dt = 0.02

self.timer = self.create\_timer(self.dt, self.timer\_callback)

```



\- 控制周期 $dt = 0.02$ 秒 = 50Hz。

\- 定时器回调 `timer\_callback` 是节点的\*\*主循环\*\*，所有状态分发与推理都在这里执行。

\- 50Hz 是腿式机器人步态控制的典型频率，与训练侧对齐。



\### 3.11 启动完成（L78）



```python

self.get\_logger().info('Inference node started')

```



日志输出确认节点初始化完成。



\---



\## 4. joint\_callback — 关节状态回调



```python

def joint\_callback(self, msg: JointState):

&#x20;   if len(msg.position) >= 10:

&#x20;       self.latest\_joint\_pos = np.array(msg.position\[:10], dtype=np.float32)

&#x20;   if len(msg.velocity) >= 10:

&#x20;       self.latest\_joint\_vel = np.array(msg.velocity\[:10], dtype=np.float32)

```



\### 功能说明



订阅 `joint\_states` 话题，接收底层编码器/驱动器上报的关节状态。



\### 数据处理逻辑



\- \*\*输入\*\*：`msg.position` 和 `msg.velocity` 是 ROS2 `JointState` 消息中的数组，包含所有关节的当前位置和速度。

\- \*\*截取前 10 个\*\*：`msg.position\[:10]` —— 假设 `bridge\_node` 按 joint\_0 到 joint\_9 的顺序发布。

\- \*\*类型转换\*\*：转为 `np.float32`，与 ONNX 模型输入类型对齐，避免推理时类型不匹配。



\### 物理/数学意义



关节位置 $q \\in \\mathbb{R}^{10}$ 和关节速度 $\\dot{q} \\in \\mathbb{R}^{10}$ 是强化学习策略 $\\pi$ 的\*\*核心观测向量\*\*：



$$

\\mathbf{o}\_{joint} = \[q\_0, q\_1, \\ldots, q\_9, \\dot{q}\_0, \\dot{q}\_1, \\ldots, \\dot{q}\_9]^T \\in \\mathbb{R}^{20}

$$



这些值描述了机器人当前的关节构型（姿态）和运动趋势，是策略网络判断"当前处于步态周期的哪个阶段"的关键信息。



\### 注意事项



\- 仅检查 `len >= 10`，不做关节名匹配。如果 `bridge\_node` 改变发布顺序，会导致\*\*静默错位\*\*——模型收到错位的角度，输出错误动作。

\- 位置单位约定为\*\*弧度（rad）\*\*，与 ONNX 模型训练时的单位一致。



\---



\## 5. imu\_callback — IMU 回调（含四元数转欧拉角推导）



```python

def imu\_callback(self, msg: Imu):

&#x20;   q = msg.orientation



&#x20;   sinr\_cosp = 2 \* (q.w \* q.x + q.y \* q.z)

&#x20;   cosr\_cosp = 1 - 2 \* (q.x \* q.x + q.y \* q.y)

&#x20;   roll = np.arctan2(sinr\_cosp, cosr\_cosp)



&#x20;   sinp = 2 \* (q.w \* q.y - q.z \* q.x)

&#x20;   if abs(sinp) >= 1:

&#x20;       pitch = np.sign(sinp) \* np.pi / 2

&#x20;   else:

&#x20;       pitch = np.arcsin(sinp)



&#x20;   siny\_cosp = 2 \* (q.w \* q.z + q.x \* q.y)

&#x20;   cosy\_cosp = 1 - 2 \* (q.y \* q.y + q.z \* q.z)

&#x20;   yaw = np.arctan2(siny\_cosp, cosy\_cosp)



&#x20;   self.latest\_euler = np.array(\[roll, pitch, yaw], dtype=np.float32)

&#x20;   self.latest\_imu\_gyro = np.array(\[msg.angular\_velocity.x, msg.angular\_velocity.y, msg.angular\_velocity.z], dtype=np.float32)

```



\### 功能说明



订阅 `imu/data` 话题，接收 IMU 上报的\*\*四元数姿态\*\*和\*\*角速度\*\*，将四元数转换为欧拉角后缓存。



\### 5.1 四元数基础



四元数 $\\mathbf{q} = \[w, x, y, z]$ 是一种表示三维旋转的方式，满足归一化约束：



$$

w^2 + x^2 + y^2 + z^2 = 1

$$



对应旋转矩阵 $R$ 为：



$$

R = \\begin{bmatrix}

1 - 2(y^2 + z^2) \& 2(xy - wz) \& 2(xz + wy) \\\\

2(xy + wz) \& 1 - 2(x^2 + z^2) \& 2(yz - wx) \\\\

2(xz - wy) \& 2(yz + wx) \& 1 - 2(x^2 + y^2)

\\end{bmatrix}

$$



\### 5.2 Roll（绕 X 轴旋转）推导



从旋转矩阵 $R$ 的元素 $R\_{31}, R\_{32}$ 提取 roll：



$$

\\sin(\\text{roll}) \\cdot \\cos(\\text{roll}) = R\_{31} = 2(xy + wz) \\quad \\Rightarrow \\quad \\sin(\\text{roll})\\cos(\\text{roll}) = 2(w x + y z)

$$



$$

\\cos^2(\\text{roll}) - \\sin^2(\\text{roll}) = R\_{32} - R\_{23}... \\quad \\Rightarrow \\quad \\cos(\\text{roll})\\cos(\\text{roll}) - ... = 1 - 2(x^2 + y^2)

$$



代码中：



```python

sinr\_cosp = 2 \* (q.w \* q.x + q.y \* q.z)   # = sin(roll) \* cos(roll)

cosr\_cosp = 1 - 2 \* (q.x \* q.x + q.y \* q.y) # = cos(roll) \* cos(roll)  \[注: 这里用的是1-2(x²+y²)]

```



$$

\\text{roll} = \\arctan2\\big(\\sin(\\text{roll})\\cos(\\text{roll}),\\ \\cos(\\text{roll})\\cos(\\text{roll})\\big) = \\arctan2(\\text{sinr\\\_cosp}, \\text{cosr\\\_cosp})

$$



> \*\*注意\*\*：$\\arctan2$ 的输入是 $\\sin\\cdot\\cos$ 和 $\\cos\\cdot\\cos$，分子分母同时除以 $\\cos(\\text{roll})$（当 $\\cos(\\text{roll}) \\neq 0$）就得到 $\\tan(\\text{roll})$，因此结果正确。用 $\\arctan2$ 而非 $\\arctan$ 是为了处理全部象限。



\### 5.3 Pitch（绕 Y 轴旋转）推导



从 $R\_{13} = 2(xz - wy)$ 和 $R\_{23} = 2(yz + wx)$... 实际上，当 roll=0, yaw=0 时：



$$

R\_{32} = 2(yz + wx) = \\sin(\\text{pitch}) \\quad \\Rightarrow \\quad \\sin(\\text{pitch}) = 2(w y - z x)

$$



代码中：



```python

sinp = 2 \* (q.w \* q.y - q.z \* q.x)  # = sin(pitch)

```



然后：



$$

\\text{pitch} = \\arcsin(\\text{sinp})

$$



\*\*Gimbal Lock 处理\*\*：



```python

if abs(sinp) >= 1:

&#x20;   pitch = np.sign(sinp) \* np.pi / 2

```



当 $|\\sin(\\text{pitch})| \\geq 1$ 时，$\\arcsin$ 会产生数值溢出（因为浮点精度，$\\sin$ 可能略大于 1）。此时 pitch 为 $\\pm 90°$（万向节锁死位置），直接钳制：



$$

\\text{pitch} = \\text{sign}(\\text{sinp}) \\times \\frac{\\pi}{2}

$$



这是四元数转欧拉角的\*\*经典数值稳定性处理\*\*。



\### 5.4 Yaw（绕 Z 轴旋转）推导



类似 roll，从 $R\_{12} = 2(xy - wz)$ 和 $R\_{11} = 1 - 2(x^2 + y^2)$... 实际推导：



$$

\\sin(\\text{yaw})\\cos(\\text{yaw}) = 2(wz + xy)

$$



$$

\\cos(\\text{yaw})\\cos(\\text{yaw}) = 1 - 2(y^2 + z^2)

$$



代码中：



```python

siny\_cosp = 2 \* (q.w \* q.z + q.x \* q.y)  # = sin(yaw) \* cos(yaw)

cosy\_cosp = 1 - 2 \* (q.y \* q.y + q.z \* q.z)  # = cos(yaw) \* cos(yaw)

```



$$

\\text{yaw} = \\arctan2(\\text{siny\\\_cosp}, \\text{cosy\\\_cosp})

$$



\### 5.5 旋转顺序约定



以上公式对应 \*\*ZYX 顺序\*\*（先 Yaw，再 Pitch，最后 Roll），也即航空航天常用的 \*\*Yaw-Pitch-Roll\*\* 约定。完整旋转关系：



$$

R = R\_z(\\text{yaw}) \\cdot R\_y(\\text{pitch}) \\cdot R\_x(\\text{roll})

$$



\### 5.6 角速度缓存



```python

self.latest\_imu\_gyro = \[msg.angular\_velocity.x, msg.angular\_velocity.y, msg.angular\_velocity.z]

```



角速度 $\\boldsymbol{\\omega} = \[\\omega\_x, \\omega\_y, \\omega\_z]^T \\in \\mathbb{R}^3$，单位 rad/s，直接从 IMU 读取，反映机器人体倾斜变化速率，是策略观测的重要组成。



\### 5.7 最终观测向量



$$

\\mathbf{o}\_{imu} = \[\\text{roll}, \\text{pitch}, \\text{yaw}, \\omega\_x, \\omega\_y, \\omega\_z]^T \\in \\mathbb{R}^6

$$



\---



\## 6. cmd\_callback — 速度指令回调



```python

def cmd\_callback(self, msg: Twist):

&#x20;   self.cmd\_vx = msg.linear.x

&#x20;   self.cmd\_vy = msg.linear.y

&#x20;   self.cmd\_dyaw = msg.angular.z

```



\### 功能说明



订阅 `cmd\_vel` 话题，接收上层（遥控器/键盘/语音助手）下发的期望速度指令。



\### 物理意义



| 变量 | 物理量 | 含义 |

|------|--------|------|

| `cmd\_vx` | 线速度 $v\_x$ (m/s) | 前后方向期望速度，正为前进 |

| `cmd\_vy` | 线速度 $v\_y$ (m/s) | 左右方向期望速度，正为左移 |

| `cmd\_dyaw` | 角速度 $\\dot{\\psi}$ (rad/s) | 偏航角速度，正为逆时针 |



这三者构成\*\*平面运动指令\*\* $\\mathbf{u} = \[v\_x, v\_y, \\dot{\\psi}]^T$，是策略网络的\*\*条件输入\*\*——告诉模型"用户想让机器人往哪个方向走、走多快"。



\### 在策略中的作用



RL 策略 $\\pi$ 实际上是一个\*\*条件策略\*\*：



$$

\\mathbf{a}\_t = \\pi(\\mathbf{o}\_t; \\mathbf{u}\_t)

$$



其中 $\\mathbf{o}\_t$ 是状态观测，$\\mathbf{u}\_t = \[v\_x, v\_y, \\dot{\\psi}]$ 是速度指令。策略根据当前姿态和期望速度，生成关节目标位置。



\---



\## 7. joy\_callback — 手柄回调（状态机驱动）



```python

def joy\_callback(self, msg: Joy):

&#x20;   if len(msg.buttons) < 3:

&#x20;       return



&#x20;   a\_pressed = msg.buttons\[0] == 1

&#x20;   b\_pressed = msg.buttons\[1] == 1



&#x20;   # A键：状态切换

&#x20;   if a\_pressed and not self.a\_pressed\_last:

&#x20;       if self.state == self.STATE\_IDLE:

&#x20;           self.start\_transition(self.target\_pos\_squat, self.STATE\_SQUAT, 2.0)

&#x20;       elif self.state == self.STATE\_SQUAT:

&#x20;           self.start\_transition(self.target\_pos\_stand, self.STATE\_STAND, 2.0)

&#x20;       elif self.state in \[self.STATE\_STAND, self.STATE\_INFERENCE]:

&#x20;           self.start\_transition(self.target\_pos\_squat, self.STATE\_SQUAT, 2.0)



&#x20;   # B键：推理开关

&#x20;   if b\_pressed and not self.b\_pressed\_last:

&#x20;       if self.state == self.STATE\_STAND:

&#x20;           self.state = self.STATE\_INFERENCE

&#x20;           self.get\_logger().info('Starting inference')

&#x20;       elif self.state == self.STATE\_INFERENCE:

&#x20;           self.start\_transition(self.target\_pos\_stand, self.STATE\_STAND, 1.0)

&#x20;           self.get\_logger().info('Stopping inference')



&#x20;   self.a\_pressed\_last = a\_pressed

&#x20;   self.b\_pressed\_last = b\_pressed

```



\### 功能说明



订阅 `joy` 话题，处理手柄按键输入，驱动状态机切换。



\### 7.1 边沿检测逻辑



核心判断：`a\_pressed and not self.a\_pressed\_last`



这是\*\*上升沿检测\*\*：只有当按键从"未按下"变为"按下"的那一刻才触发动作，按住不放不会重复触发。



数学表达：



$$

\\text{trigger}\_t = a\_t \\cdot \\overline{a\_{t-1}}

$$



其中 $a\_t \\in \\{0, 1\\}$ 是当前按键状态。这是一个\*\*离散事件的微分\*\*——在离散时间序列上，它等价于检测 $\\Delta a\_t = a\_t - a\_{t-1} = 1$。



\### 7.2 A 键状态转移



```

&#x20;        A键(2s过渡)        A键(2s过渡)

IDLE ──────────────> SQUAT ──────────────> STAND

&#x20;^                                            │

&#x20;│              A键(2s过渡)                   │

&#x20;└────────────────────────────────────────────┘

&#x20;                     ↑

&#x20;             STAND 或 INFERENCE 状态按A

&#x20;             都会回到 SQUAT

```



\- \*\*IDLE → SQUAT\*\*：上电后第一次按 A，2 秒平滑过渡到蹲姿。

\- \*\*SQUAT → STAND\*\*：蹲着再按 A，2 秒平滑过渡到站姿。

\- \*\*STAND/INFERENCE → SQUAT\*\*：站立或推理中按 A，强制回蹲（安全停止）。



\### 7.3 B 键推理开关



```

&#x20;        B键(立即)                    B键(1s过渡)

STAND ──────────────> INFERENCE ──────────────> STAND

&#x20;                    (开始RL推理)               (停止RL)

```



\- \*\*STAND → INFERENCE\*\*：站立时按 B，立即切换到推理态（不经过过渡，因为站立姿态本身已稳定，模型输出会平滑接管）。

\- \*\*INFERENCE → STAND\*\*：推理中按 B，1 秒过渡回站姿（比 A 的 2 秒快，因为这是受控退出）。



\### 7.4 安全设计要点



1\. \*\*必须先站起来才能推理\*\*：B 键只在 STAND 态生效，防止蹲着直接跑模型导致失稳。

2\. \*\*推理中按 A 优先级最高\*\*：直接回蹲，是紧急停止机制。

3\. \*\*过渡时长可配置\*\*：`start\_transition(..., duration)` 参数控制，蹲↔站 2s，推理→站 1s。



\---



\## 8. start\_transition — 状态过渡（线性插值）



```python

def start\_transition(self, target\_pos, next\_state, duration):

&#x20;   self.transition\_start\_pos = np.copy(self.latest\_joint\_pos)

&#x20;   self.transition\_target\_pos = np.copy(target\_pos)

&#x20;   self.transition\_step = self.dt / duration

&#x20;   self.transition\_progress = 0.0

&#x20;   self.next\_state = next\_state

&#x20;   self.state = self.STATE\_TRANSITIONING

```



\### 功能说明



启动一个从当前关节位置到目标位置的\*\*线性插值过渡\*\*。



\### 8.1 数学模型



设起始位置为 $\\mathbf{q}\_{start} \\in \\mathbb{R}^{10}$，目标位置为 $\\mathbf{q}\_{target} \\in \\mathbb{R}^{10}$，过渡时长 $T$ 秒，进度变量 $p \\in \[0, 1]$。



每个控制周期（dt = 0.02s），进度递增：



$$

p\_{t+1} = p\_t + \\frac{\\Delta t}{T}

$$



插值公式（在 `timer\_callback` 中执行）：



$$

\\mathbf{q}(p) = \\mathbf{q}\_{start} + (\\mathbf{q}\_{target} - \\mathbf{q}\_{start}) \\cdot p

$$



对应代码：



```python

interp = self.transition\_start\_pos + (self.transition\_target\_pos - self.transition\_start\_pos) \* self.transition\_progress

```



展开为每个关节 $i$：



$$

q\_i(p) = q\_{start,i} + (q\_{target,i} - q\_{start,i}) \\cdot p

$$



\### 8.2 过渡曲线特性



线性插值意味着关节位置随时间\*\*匀速变化\*\*，但速度在起点和终点处有\*\*突变\*\*：



$$

\\dot{q}(t) = \\frac{q\_{target} - q\_{start}}{T} \\quad (0 < t < T)

$$



$$

\\dot{q}(0^-) = 0, \\quad \\dot{q}(0^+) = \\frac{q\_{target} - q\_{start}}{T} \\quad \\text{(速度突变)}

$$



> \*\*工程含义\*\*：线性插值实现简单，但起止点有加速度冲击。更平滑的方案是用\*\*三次多项式\*\*或 \*\*S 曲线\*\*，但腿式机器人关节刚度有限，线性插值在实际中通常可接受。



\### 8.3 步长计算



```python

self.transition\_step = self.dt / duration  # = 0.02 / duration

```



\- duration = 2.0s → step = 0.01 → 每帧进度 1% → 100 帧完成

\- duration = 1.0s → step = 0.02 → 每帧进度 2% → 50 帧完成



由于 timer 是 50Hz，帧数 × 0.02s = 实际耗时，精确等于 `duration`。



\### 8.4 完成判定



```python

if self.transition\_progress >= 1.0:

&#x20;   self.transition\_progress = 1.0

&#x20;   cmd\_msg.data = self.transition\_target\_pos.tolist()

&#x20;   self.state = self.next\_state

```



当 $p \\geq 1$，锁定到目标位置并切换到 `next\_state`。这保证了终点的精确性——不会因为浮点误差停在中途。



\---



\## 9. timer\_callback — 主循环回调



```python

def timer\_callback(self):

&#x20;   if self.inference is None:

&#x20;       return



&#x20;   cmd\_msg = Float64MultiArray()



&#x20;   if self.state == self.STATE\_IDLE:

&#x20;       return

&#x20;   elif self.state == self.STATE\_SQUAT:

&#x20;       cmd\_msg.data = self.target\_pos\_squat.tolist()

&#x20;       self.motor\_cmd\_pub.publish(cmd\_msg)

&#x20;   elif self.state == self.STATE\_TRANSITIONING:

&#x20;       # 线性插值（见第8节）

&#x20;       ...

&#x20;   elif self.state == self.STATE\_STAND:

&#x20;       cmd\_msg.data = self.target\_pos\_stand.tolist()

&#x20;       self.motor\_cmd\_pub.publish(cmd\_msg)

&#x20;   elif self.state == self.STATE\_INFERENCE:

&#x20;       # 最小速度钳制 + 模型推理

&#x20;       ...

```



\### 9.1 状态分发



50Hz 定时器根据当前状态执行不同逻辑：



| 状态 | 动作 | 发布频率 |

|------|------|----------|

| IDLE | 不发指令 | — |

| SQUAT | 持续发蹲姿位置 | 50Hz |

| TRANSITIONING | 线性插值 | 50Hz |

| STAND | 持续发站姿位置 | 50Hz |

| INFERENCE | RL 模型推理 | 50Hz |



\### 9.2 最小速度钳制（INFERENCE 态）



```python

cmd\_vx = self.cmd\_vx

if abs(cmd\_vx) > 0.01 and abs(cmd\_vx) < 0.15:

&#x20;   cmd\_vx = np.sign(cmd\_vx) \* 0.15

```



数学表达：



$$

v\_x' = \\begin{cases}

\\text{sign}(v\_x) \\times 0.15 \& \\text{if } 0.01 < |v\_x| < 0.15 \\\\

v\_x \& \\text{otherwise}

\\end{cases}

$$



\*\*设计意图\*\*：



\- \*\*死区\*\*（$|v\_x| \\leq 0.01$）：小于此值视为零速，保持静止。

\- \*\*最小有效速度\*\*（$0.15$ m/s）：腿式机器人在极低速时步态难以维持稳定（单腿支撑相时间过长、重心移动不足），强制提升到 $0.15$ m/s 确保步态正常。

\- \*\*正常范围\*\*（$|v\_x| \\geq 0.15$）：原值传递。



\### 9.3 推理调用



```python

target\_q = self.inference.get\_action(

&#x20;   self.latest\_euler,        # \[roll, pitch, yaw]    ∈ R³

&#x20;   self.latest\_imu\_gyro,     # \[ωx, ωy, ωz]         ∈ R³

&#x20;   self.latest\_joint\_pos,    # \[q0..q9]             ∈ R¹⁰

&#x20;   self.latest\_joint\_vel,    # \[dq0..dq9]           ∈ R¹⁰

&#x20;   cmd\_vx,                   # 期望前进速度           ∈ R

&#x20;   self.cmd\_vy,               # 期望侧移速度           ∈ R

&#x20;   self.cmd\_dyaw              # 期望偏航角速度         ∈ R

)

```



完整观测向量：



$$

\\mathbf{o}\_t = \\underbrace{\[\\text{roll, pitch, yaw}]}\_{\\text{IMU姿态 R}^3},\\ \\underbrace{\[\\omega\_x, \\omega\_y, \\omega\_z]}\_{\\text{IMU角速度 R}^3},\\ \\underbrace{\[q\_0 \\ldots q\_9]}\_{\\text{关节位置 R}^{10}},\\ \\underbrace{\[\\dot{q}\_0 \\ldots \\dot{q}\_9]}\_{\\text{关节速度 R}^{10}},\\ \\underbrace{\[v\_x, v\_y, \\dot{\\psi}]}\_{\\text{速度指令 R}^3}

$$



总维度：$3 + 3 + 10 + 10 + 3 = 29$ 维。



策略输出：



$$

\\mathbf{a}\_t = \\pi\_\\theta(\\mathbf{o}\_t) \\in \\mathbb{R}^{10}

$$



$\\mathbf{a}\_t$ 是 10 维\*\*目标关节位置\*\*，直接发布到 `motor\_cmds` 话题，由底层 PD 控制器跟踪。



\### 9.4 异常处理



```python

try:

&#x20;   target\_q = self.inference.get\_action(...)

&#x20;   cmd\_msg.data = target\_q.tolist()

&#x20;   self.motor\_cmd\_pub.publish(cmd\_msg)

except Exception as e:

&#x20;   self.get\_logger().error(f'Error during inference: {e}')

```



推理出错时只记录日志，不崩溃——但这意味着\*\*一旦模型持续报错，机器人会失去位置指令更新\*\*，可能导致失稳。建议增加连续失败计数器，超阈值自动切回 STAND。



\---



\## 10. main() 入口函数（L209-217）



```python

def main(args=None):

&#x20;   rclpy.init(args=args)

&#x20;   node = InferenceNode()

&#x20;   rclpy.spin(node)

&#x20;   node.destroy\_node()

&#x20;   rclpy.shutdown()



if \_\_name\_\_ == '\_\_main\_\_':

&#x20;   main()

```



\### ROS2 节点标准生命周期



1\. \*\*`rclpy.init(args=args)`\*\*：初始化 ROS2 客户端库，解析命令行参数，建立与 DDS（数据分发服务）的连接。

2\. \*\*`node = InferenceNode()`\*\*：实例化节点，执行 `\_\_init\_\_` 中的所有初始化（模型加载、状态机、订阅、发布、定时器）。

3\. \*\*`rclpy.spin(node)`\*\*：进入事件循环，阻塞当前线程，持续处理回调（订阅消息到达、定时器到期）。这是节点的"心跳"——只要 spin 在运行，回调就会持续触发。

4\. \*\*`node.destroy\_node()`\*\*：销毁节点，释放所有资源（定时器、订阅、发布者）。

5\. \*\*`rclpy.shutdown()`\*\*：关闭 ROS2 客户端库，断开 DDS 连接。



\### 执行流程



```

程序启动

&#x20; │

&#x20; ▼

rclpy.init()         ← 初始化 ROS2

&#x20; │

&#x20; ▼

InferenceNode()      ← 加载模型/状态机/订阅/发布/定时器

&#x20; │

&#x20; ▼

rclpy.spin(node)     ← 阻塞，持续触发回调

&#x20; │                    │

&#x20; │                    ├─> joint\_callback (传感器数据到达时)

&#x20; │                    ├─> imu\_callback   (传感器数据到达时)

&#x20; │                    ├─> cmd\_callback   (指令到达时)

&#x20; │                    ├─> joy\_callback   (手柄数据到达时)

&#x20; │                    └─> timer\_callback (每 20ms 一次)

&#x20; │

&#x20; │  (Ctrl+C 或 kill 信号)

&#x20; ▼

destroy\_node()       ← 清理资源

&#x20; │

&#x20; ▼

rclpy.shutdown()     ← 关闭 ROS2

&#x20; │

&#x20; ▼

程序退出

```



\---



\## 11. 附录：数学公式汇总



\### 11.1 四元数转欧拉角（ZYX 顺序）



$$

\\text{roll} = \\arctan2\\big(2(wx + yz),\\ 1 - 2(x^2 + y^2)\\big)

$$



$$

\\text{pitch} = \\arcsin\\big(2(wy - zx)\\big), \\quad \\text{钳制到 } \[-\\frac{\\pi}{2}, \\frac{\\pi}{2}]

$$



$$

\\text{yaw} = \\arctan2\\big(2(wz + xy),\\ 1 - 2(y^2 + z^2)\\big)

$$



\### 11.2 线性插值



$$

\\mathbf{q}(p) = \\mathbf{q}\_{start} + (\\mathbf{q}\_{target} - \\mathbf{q}\_{start}) \\cdot p, \\quad p \\in \[0, 1]

$$



$$

p\_{t+1} = p\_t + \\frac{\\Delta t}{T}

$$



\### 11.3 最小速度钳制



$$

v\_x' = \\begin{cases}

\\text{sign}(v\_x) \\times 0.15 \& 0.01 < |v\_x| < 0.15 \\\\

v\_x \& \\text{otherwise}

\\end{cases}

$$



\### 11.4 上升沿检测



$$

\\text{trigger}\_t = a\_t \\cdot (1 - a\_{t-1}), \\quad a\_t \\in \\{0, 1\\}

$$



\### 11.5 PD 控制律（底层，本节点不可见）



$$

\\tau = K\_p (\\mathbf{q}\_{target} - \\mathbf{q}) - K\_d \\dot{\\mathbf{q}}

$$



\### 11.6 策略函数



$$

\\mathbf{a}\_t = \\pi\_\\theta(\\mathbf{o}\_t), \\quad \\mathbf{o}\_t \\in \\mathbb{R}^{29}, \\quad \\mathbf{a}\_t \\in \\mathbb{R}^{10}

$$



\---



\## 12. 观测向量维度总结



| 观测分量 | 维度 | 来源回调 |

|----------|------|----------|

| 欧拉角 \[roll, pitch, yaw] | 3 | `imu\_callback` |

| IMU 角速度 \[ωx, ωy, ωz] | 3 | `imu\_callback` |

| 关节位置 \[q0..q9] | 10 | `joint\_callback` |

| 关节速度 \[dq0..dq9] | 10 | `joint\_callback` |

| 速度指令 \[vx, vy, dyaw] | 3 | `cmd\_callback` |

| \*\*总计\*\* | \*\*29\*\* | — |



输出：10 维目标关节位置 → `motor\_cmds` 话题（50Hz）。



\---



\## 13. 设计要点与潜在风险



\### 13.1 设计亮点



1\. \*\*解耦架构\*\*：RL 只负责策略输出位置，PD/力控在底层，分层清晰。

2\. \*\*安全门控\*\*：状态机 + 手柄双键，保证模型不会在上电即跑。

3\. \*\*平滑过渡\*\*：所有姿态切换走线性插值，防冲击。

4\. \*\*最小速度钳制\*\*：规避低速不稳定区。

5\. \*\*观测组装标准\*\*：IMU+关节状态+速度指令，与训练侧对齐。



\### 13.2 潜在风险



1\. \*\*关节顺序假设\*\*：L83 `msg.position\[:10]` 假设顺序固定，若 bridge 改了顺序会静默错位。建议增加关节名校验。

2\. \*\*模型加载失败"假活"\*\*：L27 置 `None` 后节点继续运行但不输出，建议失败直接 `sys.exit`。

3\. \*\*IDLE 态无指令\*\*：`timer\_callback` 在 IDLE 时直接 return，不 publish 任何 motor\_cmds——若底层默认行为不是"保持上电位置"，机器人可能松力。需确认底层驱动在无指令时的行为。

4\. \*\*推理异常无降级\*\*：L206-207 出错只 log 不崩，连续失败会导致指令停止更新，建议增加失败计数与自动回退。

5\. \*\*线性插值加速度突变\*\*：起止点有速度阶跃，对高刚度关节可能产生冲击，可考虑 S 曲线优化。



